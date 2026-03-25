"""
SmartDoc — RapidCanvas DataApp  (single-file, zero local imports)
Author: Vanessa Hu | Georgia Tech MS Analytics Practicum, Spring 2026

Upload ONLY this file + requirements.txt to RapidCanvas DataApps.
Set env var: ANTHROPIC_API_KEY

All pipeline code is inlined here:
  • ClaudeExtractor  — Claude API extraction
  • Validator        — 3-layer rule-based validation
  • CorrectionLog    — human preference recorder
  • RuleDistiller    — pattern learning engine
  • build_export_pdf — branded PDF export (reportlab)
"""

# ── Standard imports ──────────────────────────────────────────────────────────
from __future__ import annotations

import io
import json
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Union

import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent
MEMORY_DIR = _ROOT / "data" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# Sample invoice: place a PDF named SAMPLE_invoice.pdf next to this file (optional)
SAMPLE_PDF = _ROOT / "SAMPLE_invoice.pdf"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartDoc",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: SmartDoc dark / gold brand ──────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #1a1f26; color: #e8e4d8; }
  section[data-testid="stSidebar"] { background-color: #12161c; }

  .sd-title   { font-size:2.6rem; font-weight:800; color:#c9a84c;
                letter-spacing:-0.5px; margin:0; line-height:1.1; }
  .sd-tagline { font-size:0.95rem; color:#7a7060; font-style:italic; margin:0.2rem 0 0 0; }
  .sd-divider { border:none; border-top:1px solid #2a3040; margin:1rem 0; }
  .step-lbl   { font-size:0.7rem; font-weight:700; letter-spacing:1.8px;
                color:#c9a84c; text-transform:uppercase; margin:1rem 0 0.4rem 0; }

  .badge { display:inline-block; padding:4px 13px; border-radius:16px;
           font-size:0.8rem; font-weight:700; }
  .b-nt   { background:#1b3a1e; color:#81c784; border:1px solid #388e3c; }
  .b-ok   { background:#1a2a40; color:#90caf9; border:1px solid #1565c0; }
  .b-warn { background:#3a1f0a; color:#ffb74d; border:1px solid #e65100; }

  .mem-card { background:#1e2530; border-left:3px solid #c9a84c;
              padding:0.7rem 0.9rem; border-radius:5px; margin:0.35rem 0; }
  .mem-num  { font-size:1.7rem; font-weight:800; color:#c9a84c; line-height:1; }
  .mem-lbl  { font-size:0.68rem; color:#5a6070; text-transform:uppercase; letter-spacing:0.7px; }

  .rule-pill { background:#2a1f0a; color:#ffd580; border:1px solid #7a5c10;
               border-radius:3px; padding:1px 6px; font-size:0.72rem; font-family:monospace; }

  .dot { display:inline-block; width:9px; height:9px; border-radius:50%;
         margin-right:4px; vertical-align:middle; }
  .d-hi  { background:#4caf50; }
  .d-med { background:#ffc107; }
  .d-lo  { background:#f44336; }
  .d-mis { background:#444; }

  .stTextInput > label { color:#c9a84c !important; font-size:0.75rem !important;
    font-weight:600 !important; letter-spacing:0.7px !important; text-transform:uppercase !important; }
  .stFileUploader > label { color:#c9a84c !important; }
  .stButton > button[kind="primary"] {
    background:#c9a84c !important; color:#1a1f26 !important;
    font-weight:700 !important; border:none !important; }
  .stButton > button[kind="primary"]:hover { background:#dfc060 !important; }
  .stDownloadButton > button {
    background:#1b3a1e !important; color:#81c784 !important;
    border:1px solid #388e3c !important; font-weight:700 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — EXTRACTION  (claude_extractor.py inlined)
# ══════════════════════════════════════════════════════════════════════════════

TARGET_FIELDS = [
    "NUMBER", "DATE", "DUE_DATE", "SELLER_NAME",
    "BUYER", "BILL_TO", "SEND_TO", "TOTAL", "TOTAL_WORDS",
]
REQUIRED_FIELDS = {"NUMBER", "DATE", "SELLER_NAME", "TOTAL"}


@dataclass
class ExtractionResult:
    pdf_path:      str
    fields:        dict
    confidence:    dict
    raw_text:      str
    validation:    object = None
    rules_applied: list   = dc_field(default_factory=list)
    model:         str    = "claude-opus-4-6"
    pages:         int    = 0

    @property
    def is_valid(self):
        return self.validation.is_valid if self.validation else False

    @property
    def no_touch(self):
        return self.validation.no_touch if self.validation else False

    def to_dict(self):
        return {
            "pdf_path": self.pdf_path, "fields": self.fields,
            "confidence": self.confidence, "is_valid": self.is_valid,
            "no_touch": self.no_touch, "rules_applied": self.rules_applied,
            "pages": self.pages, "model": self.model,
        }


class ClaudeExtractor:
    SYSTEM_PROMPT = """You are an expert invoice data extraction system.
Return ONLY a valid JSON object — no explanation, no markdown, no code blocks.
If a field is not present, use an empty string "".
Be conservative: only extract values you are confident about."""

    EXTRACTION_PROMPT = """Extract the following fields from this invoice text.

Fields:
- NUMBER: invoice number or ID
- DATE: invoice issue date
- DUE_DATE: payment due date if present
- SELLER_NAME: name of the business issuing the invoice
- BUYER: name of the client being billed
- BILL_TO: billing address as a single string
- SEND_TO: shipping address if different from billing
- TOTAL: final total amount as a number (e.g. "1250.00")
- TOTAL_WORDS: total written in words if present

Also provide confidence per field: "high", "medium", "low", or "missing".

Return this JSON structure:
{{
  "fields": {{
    "NUMBER": "...", "DATE": "...", "DUE_DATE": "...",
    "SELLER_NAME": "...", "BUYER": "...", "BILL_TO": "...",
    "SEND_TO": "...", "TOTAL": "...", "TOTAL_WORDS": "..."
  }},
  "confidence": {{
    "NUMBER": "high", "DATE": "high", "DUE_DATE": "missing",
    "SELLER_NAME": "high", "BUYER": "medium", "BILL_TO": "high",
    "SEND_TO": "missing", "TOTAL": "high", "TOTAL_WORDS": "missing"
  }}
}}

Invoice text:
---
{invoice_text}
---"""

    def __init__(self, api_key, model="claude-opus-4-6",
                 validator=None, distiller=None, correction_log=None):
        import anthropic
        self.client        = anthropic.Anthropic(api_key=api_key)
        self.model         = model
        self.validator     = validator
        self.distiller     = distiller
        self.correction_log = correction_log

    def process_pdf(self, pdf_path):
        import pdfplumber
        pdf_path = Path(pdf_path)
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"[Page {i+1}]\n{text}")
                for table in page.extract_tables():
                    for row in table:
                        row_text = " | ".join(str(c).strip() if c else "" for c in row)
                        if row_text.strip(" |"):
                            pages_text.append(row_text)

        raw_text = "\n".join(pages_text)
        if len(raw_text) > 6000:
            raw_text = raw_text[:6000] + "\n[... truncated ...]"

        fields, confidence = self._call_claude(raw_text)

        rules_applied = []
        if self.distiller:
            fields, rules_applied = self.distiller.apply(fields)

        validation = self.validator.validate(fields) if self.validator else None

        return ExtractionResult(
            pdf_path=str(pdf_path), fields=fields, confidence=confidence,
            raw_text=raw_text, validation=validation, rules_applied=rules_applied,
            model=self.model, pages=n_pages,
        )

    def _call_claude(self, invoice_text):
        prompt = self.EXTRACTION_PROMPT.format(invoice_text=invoice_text)
        msg = self.client.messages.create(
            model=self.model, max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_response(msg.content[0].text.strip())

    def _parse_response(self, text):
        clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return {f: "" for f in TARGET_FIELDS}, {f: "missing" for f in TARGET_FIELDS}
        try:
            data       = json.loads(match.group())
            fields     = data.get("fields", {})
            confidence = data.get("confidence", {})
            for f in TARGET_FIELDS:
                fields.setdefault(f, "")
                confidence.setdefault(f, "missing" if not fields.get(f) else "medium")
            return {k: str(v).strip() for k, v in fields.items()}, confidence
        except json.JSONDecodeError:
            return {f: "" for f in TARGET_FIELDS}, {f: "missing" for f in TARGET_FIELDS}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — VALIDATION  (validator.py inlined)
# ══════════════════════════════════════════════════════════════════════════════

_DATE_PATS = [
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}",
    r"\d{4}[./-]\d{1,2}[./-]\d{1,2}",
    r"\d{1,2}\s+\w+\s+\d{4}",
    r"\w+\s+\d{1,2},?\s+\d{4}",
]
_NUM_PATS = [r"[A-Z0-9]{3,}-?\d+", r"#?\d{4,}", r"[A-Z]{2,}\d{4,}"]
_AMT_PAT  = re.compile(r"[\$€£₺]?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?")


def _parse_date_flexible(s):
    fmts = ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y",
            "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
            "%d/%m/%y", "%m/%d/%y"]
    for fmt in fmts:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _extract_numeric(s):
    if not s:
        return None
    cleaned = re.sub(r"[^\d.,]", "", s.strip())
    if re.match(r"^\d{1,3}(\.\d{3})+(,\d+)?$", cleaned):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


@dataclass
class ValidationIssue:
    field:   str
    level:   str   # "error" | "warning" | "anomaly"
    message: str
    value:   str = ""


@dataclass
class ValidationResult:
    fields:   dict
    issues:   list = dc_field(default_factory=list)
    is_valid: bool = True
    no_touch: bool = False

    def add(self, issue):
        self.issues.append(issue)
        if issue.level == "error":
            self.is_valid = False


class Validator:
    def __init__(self, anomaly_z_threshold=2.5, required_fields=None):
        self.z_thresh        = anomaly_z_threshold
        self.required_fields = required_fields or ["DATE", "NUMBER", "TOTAL"]

    def validate(self, fields, history=None):
        result = ValidationResult(fields=fields)
        self._completeness(fields, result)
        self._formats(fields, result)
        self._logic(fields, result)
        if history:
            self._anomalies(fields, history, result)
        result.no_touch = len(result.issues) == 0
        return result

    def _completeness(self, fields, result):
        for req in self.required_fields:
            if not fields.get(req, "").strip():
                result.add(ValidationIssue(field=req, level="error",
                                           message="Required field missing or empty"))

    def _formats(self, fields, result):
        for fname, value in fields.items():
            v = (value or "").strip()
            if not v:
                continue
            if fname in ("DATE", "DUE_DATE"):
                if not any(re.search(p, v) for p in _DATE_PATS):
                    result.add(ValidationIssue(field=fname, level="warning",
                        message="Does not match a known date format", value=v))
            elif fname == "NUMBER":
                if not any(re.search(p, v) for p in _NUM_PATS):
                    result.add(ValidationIssue(field=fname, level="warning",
                        message="Invoice number format looks unusual", value=v))
            elif fname == "TOTAL":
                if not _AMT_PAT.search(v):
                    result.add(ValidationIssue(field=fname, level="warning",
                        message="Does not look like a monetary amount", value=v))

    def _logic(self, fields, result):
        d1 = _parse_date_flexible(fields.get("DATE", ""))
        d2 = _parse_date_flexible(fields.get("DUE_DATE", ""))
        if d1 and d2 and d2 < d1:
            result.add(ValidationIssue(field="DUE_DATE", level="error",
                message=f"Due date is before issue date"))
        total_val = _extract_numeric(fields.get("TOTAL", ""))
        if total_val is not None and total_val <= 0:
            result.add(ValidationIssue(field="TOTAL", level="error",
                message="Total must be positive", value=fields.get("TOTAL", "")))
        bill = fields.get("BILL_TO", "").strip()
        send = fields.get("SEND_TO", "").strip()
        if bill and send and bill == send and len(bill) > 5:
            result.add(ValidationIssue(field="BILL_TO/SEND_TO", level="warning",
                message="Bill-to and ship-to addresses are identical"))

    def _anomalies(self, fields, history, result):
        for fname in ["TOTAL"]:
            cur = _extract_numeric(fields.get(fname, ""))
            if cur is None:
                continue
            past = [_extract_numeric(h.get(fname, "")) for h in history]
            past = [v for v in past if v is not None]
            if len(past) < 5:
                continue
            mean = statistics.mean(past)
            std  = statistics.stdev(past)
            if std == 0:
                continue
            z = abs(cur - mean) / std
            if z > self.z_thresh:
                direction = "high" if cur > mean else "low"
                result.add(ValidationIssue(field=fname, level="anomaly",
                    message=f"Unusually {direction} (z={z:.1f}, mean={mean:.2f})"))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CORRECTION LOG  (correction_log.py inlined)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CorrectionEntry:
    doc_id:      str
    template_id: str
    field:       str
    original:    str
    corrected:   str
    timestamp:   str
    confidence:  Optional[float] = None
    note:        Optional[str]   = None

    @property
    def is_meaningful(self):
        return self.original.strip() != self.corrected.strip()

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class CorrectionLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def record(self, doc_id, field, original, corrected,
               template_id="unknown", confidence=None, note=None,
               *, only_if_changed=True):
        if only_if_changed and original.strip() == corrected.strip():
            return None
        entry = CorrectionEntry(
            doc_id=doc_id, template_id=template_id, field=field,
            original=original, corrected=corrected,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=confidence, note=note,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")
        return entry

    def entries(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield CorrectionEntry.from_dict(json.loads(line))
                    except (json.JSONDecodeError, TypeError):
                        continue

    def load_all(self):
        return list(self.entries())

    def stats(self):
        all_e = self.load_all()
        return {
            "total":      len(all_e),
            "by_field":   dict(Counter(e.field for e in all_e).most_common()),
            "unique_docs": len({e.doc_id for e in all_e}),
        }

    def __len__(self):
        return sum(1 for _ in self.entries())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — RULE DISTILLER  (rule_distiller.py inlined)
# ══════════════════════════════════════════════════════════════════════════════

_DATE_RULES = [
    (r"^(\d{2})-(\d{2})-(\d{2})$",   r"20\3-\2-\1"),
    (r"^(\d{2})/(\d{2})/(\d{4})$",   r"\3-\2-\1"),
    (r"^(\d{2})\.(\d{2})\.(\d{4})$", r"\3-\2-\1"),
    (r"^(\d{4})(\d{2})(\d{2})$",     r"\1-\2-\3"),
]
_AMT_RULES = [
    (r"[$€£]\s*", ""),
    (r"(\d),(\d{3})", r"\1\2"),
]


@dataclass
class FormatRule:
    rule_type:   str = "format"
    field:       str = ""
    pattern:     str = ""
    replacement: str = ""
    confidence:  float = 0.0
    examples:    list = dc_field(default_factory=list)

    def apply(self, value):
        new = re.sub(self.pattern, self.replacement, value)
        return new, new != value


@dataclass
class SubstitutionRule:
    rule_type:  str = "substitution"
    field:      str = ""
    wrong:      str = ""
    correct:    str = ""
    frequency:  int = 0
    confidence: float = 0.0

    def apply(self, value):
        if value.strip() == self.wrong.strip():
            return self.correct, True
        return value, False


@dataclass
class AnomalyThresholdRule:
    rule_type: str = "anomaly_threshold"
    field:     str = "TOTAL"
    mean:      float = 0.0
    std:       float = 0.0
    k:         float = 3.0
    n_samples: int = 0

    @property
    def lower(self):
        return max(0.0, self.mean - self.k * self.std)

    @property
    def upper(self):
        return self.mean + self.k * self.std

    def is_anomaly(self, value):
        return value < self.lower or value > self.upper


class RuleDistiller:
    def __init__(self, log, rules_path="data/memory/rules.json",
                 min_freq=3, min_conf=0.60):
        self.log        = log
        self.rules_path = Path(rules_path)
        self.min_freq   = min_freq
        self.min_conf   = min_conf
        self.rules      = []
        self._loaded    = False

    def fit(self):
        entries = self.log.load_all()
        if not entries:
            self.rules = []
            self._save()
            return self
        by_field = defaultdict(list)
        for e in entries:
            if e.is_meaningful:
                by_field[e.field].append(e)
        rules = []
        for fld, fentries in by_field.items():
            rules.extend(self._substitution_rules(fld, fentries))
            rules.extend(self._format_rules(fld, fentries))
        total_entries = by_field.get("TOTAL", [])
        ar = self._anomaly_rule(total_entries)
        if ar:
            rules.append(ar)
        self.rules   = rules
        self._loaded = True
        self._save()
        return self

    def _substitution_rules(self, field, entries):
        pair_counts = Counter((e.original.strip(), e.corrected.strip()) for e in entries)
        total = len(entries)
        rules = []
        for (wrong, correct), count in pair_counts.most_common():
            if count < self.min_freq:
                break
            conf = count / total
            if conf >= self.min_conf:
                rules.append(SubstitutionRule(field=field, wrong=wrong, correct=correct,
                                              frequency=count, confidence=round(conf, 4)))
        return rules

    def _format_rules(self, field, entries):
        patterns = _DATE_RULES if field in ("DATE", "DUE_DATE") else \
                   _AMT_RULES  if field == "TOTAL" else []
        rules = []
        total = len(entries)
        for pat, repl in patterns:
            matched = 0
            examples = []
            for e in entries:
                try:
                    t = re.sub(pat, repl, e.original.strip())
                    if t == e.corrected.strip() and t != e.original.strip():
                        matched += 1
                        if len(examples) < 3:
                            examples.append(f"{e.original!r} → {e.corrected!r}")
                except re.error:
                    continue
            if matched and matched / total >= self.min_conf:
                rules.append(FormatRule(field=field, pattern=pat, replacement=repl,
                                        confidence=round(matched / total, 4),
                                        examples=examples))
        return rules

    def _anomaly_rule(self, total_entries):
        values = []
        for e in total_entries:
            try:
                v = float(re.sub(r"[^\d.]", "", e.corrected))
                if v > 0:
                    values.append(v)
            except ValueError:
                continue
        if len(values) < 5:
            return None
        mean = statistics.mean(values)
        std  = statistics.stdev(values) if len(values) > 1 else 0.0
        return AnomalyThresholdRule(field="TOTAL", mean=round(mean, 2),
                                    std=round(std, 2), k=3.0, n_samples=len(values))

    def apply(self, fields):
        if not self._loaded:
            self.load()
        updated = dict(fields)
        applied = []
        for rule in self.rules:
            if rule.rule_type == "anomaly_threshold":
                continue
            fld = rule.field
            if fld not in updated:
                continue
            new_val, changed = rule.apply(updated[fld])
            if changed:
                applied.append({"field": fld, "rule_type": rule.rule_type,
                                 "before": updated[fld], "after": new_val,
                                 "confidence": getattr(rule, "confidence", None)})
                updated[fld] = new_val
        return updated, applied

    def _save(self):
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        self.rules_path.write_text(
            json.dumps([asdict(r) for r in self.rules], indent=2, ensure_ascii=False)
        )

    def load(self):
        if not self.rules_path.exists():
            self.rules = []
            return self
        raw = json.loads(self.rules_path.read_text())
        self.rules = []
        for d in raw:
            rt = d.get("rule_type")
            if rt == "format":
                self.rules.append(FormatRule(**d))
            elif rt == "substitution":
                self.rules.append(SubstitutionRule(**d))
            elif rt == "anomaly_threshold":
                self.rules.append(AnomalyThresholdRule(**d))
        self._loaded = True
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PDF EXPORT  (reportlab)
# ══════════════════════════════════════════════════════════════════════════════

FIELD_LABELS = {
    "NUMBER": "Invoice Number", "DATE": "Invoice Date", "DUE_DATE": "Due Date",
    "SELLER_NAME": "Seller / Vendor", "BUYER": "Buyer / Client",
    "BILL_TO": "Bill-To Address", "SEND_TO": "Ship-To Address",
    "TOTAL": "Total Amount", "TOTAL_WORDS": "Total in Words",
}


def build_export_pdf(fields, confidence, validation, rules_applied,
                     invoice_name, model) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch,  bottomMargin=0.75*inch)

    GOLD  = colors.HexColor("#c9a84c")
    DARK  = colors.HexColor("#1a1f26")
    MID   = colors.HexColor("#2a3040")
    LIGHT = colors.HexColor("#e8e4d8")
    GREEN = colors.HexColor("#4caf50")
    AMBER = colors.HexColor("#ffc107")
    RED   = colors.HexColor("#f44336")
    GREY  = colors.HexColor("#6b7280")

    base = getSampleStyleSheet()
    def sty(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    story = []
    story.append(Paragraph("SmartDoc",
                            sty("T", fontSize=22, textColor=GOLD,
                                fontName="Helvetica-Bold", spaceAfter=2)))
    story.append(Paragraph("Your Business Memory. Smarter Every Invoice.",
                            sty("G", fontSize=9, textColor=GREY,
                                fontStyle="italic", spaceAfter=8)))
    story.append(HRFlowable(width="100%", thickness=1, color=GOLD, spaceAfter=6))
    story.append(Paragraph(
        f"<b>Source:</b> {invoice_name or 'unknown'}  &nbsp;|&nbsp; "
        f"<b>Processed:</b> {datetime.now().strftime('%B %d, %Y  %H:%M')}  &nbsp;|&nbsp; "
        f"<b>Model:</b> {model}",
        sty("M", fontSize=7.5, textColor=GREY),
    ))
    story.append(Spacer(1, 10))

    # Status badge
    story.append(Paragraph("EXTRACTION STATUS",
                            sty("S", fontSize=9, textColor=GOLD,
                                fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)))
    if validation:
        if validation.no_touch:
            stxt, scol = "NO-TOUCH — All fields extracted and validated", GREEN
        elif validation.is_valid:
            stxt, scol = "VALID — Minor warnings only", AMBER
        else:
            stxt, scol = "NEEDS REVIEW — Errors detected", RED
    else:
        stxt, scol = "NOT VALIDATED", GREY

    badge = Table([[Paragraph(f"  {stxt}  ",
                              sty("B", fontSize=9, fontName="Helvetica-Bold",
                                  textColor=colors.white))]],
                  colWidths=[4.5*inch])
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), scol),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
    ]))
    story.append(badge)

    if validation and validation.issues:
        story.append(Spacer(1, 6))
        for iss in validation.issues:
            c = RED if iss.level == "error" else AMBER
            story.append(Paragraph(
                f"<b>[{'ERROR' if iss.level=='error' else 'WARNING'}]</b>  "
                f"{FIELD_LABELS.get(iss.field, iss.field)}: {iss.message}",
                sty("I", fontSize=8, textColor=c, leftIndent=8, spaceBefore=2),
            ))

    # Fields table
    story.append(Paragraph("EXTRACTED FIELDS",
                            sty("S2", fontSize=9, textColor=GOLD,
                                fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)))
    CONF_C = {"high": GREEN, "medium": AMBER, "low": RED, "missing": GREY}
    rows = [[
        Paragraph("<b>Field</b>",      sty("H", fontSize=8, fontName="Helvetica-Bold", textColor=LIGHT)),
        Paragraph("<b>Value</b>",      sty("H", fontSize=8, fontName="Helvetica-Bold", textColor=LIGHT)),
        Paragraph("<b>Confidence</b>", sty("H", fontSize=8, fontName="Helvetica-Bold", textColor=LIGHT)),
    ]]
    for fld in TARGET_FIELDS:
        val  = fields.get(fld, "") or "—"
        conf = confidence.get(fld, "missing")
        req  = " *" if fld in REQUIRED_FIELDS else ""
        rows.append([
            Paragraph(f"{FIELD_LABELS.get(fld,fld)}{req}", sty("TC", fontSize=8, textColor=LIGHT)),
            Paragraph(str(val),                             sty("TV", fontSize=8, textColor=LIGHT)),
            Paragraph(conf.upper(), sty("TK", fontSize=7.5, fontName="Helvetica-Bold",
                                        textColor=CONF_C.get(conf, GREY))),
        ])
    ftbl = Table(rows, colWidths=[2.0*inch, 3.6*inch, 1.2*inch])
    ftbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),  MID),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [DARK, colors.HexColor("#1f2530")]),
        ("GRID",           (0,0),(-1,-1), 0.4, colors.HexColor("#2a3040")),
        ("TOPPADDING",     (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 6),
        ("LEFTPADDING",    (0,0),(-1,-1), 8),
        ("VALIGN",         (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(ftbl)
    story.append(Spacer(1, 4))
    story.append(Paragraph("* Required field", sty("FN", fontSize=7, textColor=GREY)))

    if rules_applied:
        story.append(Paragraph("BUSINESS MEMORY — AUTO-CORRECTIONS APPLIED",
                                sty("S3", fontSize=9, textColor=GOLD,
                                    fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)))
        for r in rules_applied:
            story.append(Paragraph(
                f"<b>{r['field']}</b>: \"{r.get('before') or '(empty)'}\" → \"{r['after']}\" "
                f"<font color='grey'>({r['rule_type']})</font>",
                sty("R", fontSize=8, textColor=LIGHT, leftIndent=10, spaceBefore=3),
            ))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "SmartDoc  ·  AI-Assisted Invoice Automation  ·  "
        "Georgia Tech MS Analytics Practicum, Spring 2026  ·  Powered by Claude API",
        sty("F", fontSize=7, textColor=GREY, alignment=1),
    ))
    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — MEMORY SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _get_memory():
    log  = CorrectionLog(MEMORY_DIR / "corrections.jsonl")
    dist = RuleDistiller(log, rules_path=MEMORY_DIR / "rules.json")
    if (MEMORY_DIR / "rules.json").exists():
        dist.load()
    return log, dist


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

CONF_META = {
    "high":    "d-hi",
    "medium":  "d-med",
    "low":     "d-lo",
    "missing": "d-mis",
}

for _k, _v in {
    "result": None, "edits": {}, "confirmed": False,
    "pdf_bytes": None, "invoice_name": "",
    "sess_processed": 0, "sess_notouch": 0,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Business Memory
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 Business Memory")
    st.caption("SmartDoc learns from every correction you make.")
    st.markdown("")
    log, dist = _get_memory()
    stats  = log.stats()
    n_proc = st.session_state.sess_processed
    n_nt   = st.session_state.sess_notouch
    nt_pct = f"{100*n_nt//max(n_proc,1)}%" if n_proc else "—"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="mem-card">
          <div class="mem-num">{len(dist.rules)}</div>
          <div class="mem-lbl">Rules Learned</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="mem-card">
          <div class="mem-num">{stats["total"]}</div>
          <div class="mem-lbl">Corrections</div></div>""", unsafe_allow_html=True)

    st.metric("Session Invoices", n_proc)
    st.metric("No-Touch Rate",    nt_pct)

    if dist.rules:
        st.markdown("---")
        st.markdown("**Active Rules**")
        for rule in dist.rules:
            fld = getattr(rule, "field", "?")
            if rule.rule_type == "substitution":
                st.markdown(f'<span class="rule-pill">{fld}</span> `"{rule.wrong}"` → `"{rule.correct}"`',
                            unsafe_allow_html=True)
            elif rule.rule_type == "format":
                st.markdown(f'<span class="rule-pill">{fld}</span> format rule ({rule.confidence:.0%})',
                            unsafe_allow_html=True)
            elif rule.rule_type == "anomaly_threshold":
                st.markdown(f'<span class="rule-pill">TOTAL</span> anomaly: {rule.lower:.0f}–{rule.upper:.0f}',
                            unsafe_allow_html=True)
    else:
        st.info("Process & correct invoices to build memory.", icon="💡")

    st.markdown("---")
    st.caption("SmartDoc · Georgia Tech · Spring 2026")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:1rem 0 0.2rem 0;">
  <div class="sd-title">SmartDoc</div>
  <div class="sd-tagline">Your Business Memory. Smarter Every Invoice.</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<hr class="sd-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load Invoice
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-lbl">Step 1 — Load Invoice</div>', unsafe_allow_html=True)

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    with st.expander("⚙️ API Key", expanded=True):
        api_key = st.text_input("Paste your model API key here — we will never save it.",
                                type="password",
                                placeholder="sk-ant-api03-...",
                                help="Stays in memory only — never stored to disk.")

uploaded = st.file_uploader("Upload your invoice PDF", type=["pdf"])

pdf_bytes_to_use = None
pdf_name_to_use  = ""
if uploaded:
    pdf_bytes_to_use = uploaded.getvalue()
    pdf_name_to_use  = uploaded.name

if pdf_bytes_to_use and api_key:
    if st.button("🔍  Extract Fields", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes_to_use)
            tmp_path = tmp.name
        with st.spinner("SmartDoc is reading your invoice…"):
            try:
                log, dist = _get_memory()
                result = ClaudeExtractor(
                    api_key=api_key, model="claude-opus-4-6",
                    validator=Validator(), distiller=dist, correction_log=log,
                ).process_pdf(tmp_path)
                st.session_state.result       = result
                st.session_state.edits        = dict(result.fields)
                st.session_state.confirmed    = False
                st.session_state.pdf_bytes    = None
                st.session_state.invoice_name = pdf_name_to_use
                st.session_state.sess_processed += 1
                if result.no_touch:
                    st.session_state.sess_notouch += 1
            except Exception as e:
                st.error(f"Extraction failed: {e}", icon="🚨")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        st.rerun()
elif pdf_bytes_to_use and not api_key:
    st.warning("Paste your API key above to continue.", icon="🔑")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Review & Correct
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.result:
    result = st.session_state.result
    st.markdown('<hr class="sd-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-lbl">Step 2 — Review &amp; Correct</div>',
                unsafe_allow_html=True)

    if result.no_touch:
        badge = '<span class="badge b-nt">✓  NO-TOUCH — All fields validated</span>'
    elif result.is_valid:
        badge = '<span class="badge b-ok">✓  VALID — Minor warnings</span>'
    else:
        badge = '<span class="badge b-warn">⚠  NEEDS REVIEW — Errors detected</span>'
    st.markdown(badge, unsafe_allow_html=True)
    st.markdown("")

    if result.rules_applied:
        names = ", ".join(f"`{r['field']}`" for r in result.rules_applied)
        st.info(f"🧠 Business Memory auto-corrected {len(result.rules_applied)} "
                f"field(s) before display: {names}", icon="🧠")

    col_fields, col_info = st.columns([3, 2])

    with col_fields:
        st.markdown("**Edit any field that looks wrong:**")
        new_edits = {}
        for fld in TARGET_FIELDS:
            conf    = result.confidence.get(fld, "missing")
            dot_cls = CONF_META.get(conf, "d-mis")
            c_dot, c_inp = st.columns([0.5, 9.5])
            with c_dot:
                st.markdown(f'<div style="margin-top:30px;text-align:center;">'
                            f'<span class="dot {dot_cls}"></span></div>',
                            unsafe_allow_html=True)
            with c_inp:
                cur = st.session_state.edits.get(fld, "")
                new_edits[fld] = st.text_input(
                    f"{FIELD_LABELS.get(fld,fld)}{'  ★' if fld in REQUIRED_FIELDS else ''}",
                    value=cur, key=f"ei_{fld}",
                    placeholder="(not found)" if not cur else "",
                    disabled=st.session_state.confirmed,
                )
        st.session_state.edits = new_edits

    with col_info:
        st.markdown("**Confidence**")
        st.markdown("""
        <div style="font-size:0.8rem;line-height:2.2;">
          <span class="dot d-hi"></span>HIGH — extracted with confidence<br>
          <span class="dot d-med"></span>MED — verify recommended<br>
          <span class="dot d-lo"></span>LOW — please review carefully<br>
          <span class="dot d-mis"></span>— &nbsp;not found in document
        </div>""", unsafe_allow_html=True)
        st.caption("★ Required field")
        st.markdown("")

        if result.validation and result.validation.issues:
            st.markdown("**Validation Issues**")
            for iss in result.validation.issues:
                icon = "🔴" if iss.level == "error" else "⚠️"
                st.markdown(f"{icon} **{FIELD_LABELS.get(iss.field,iss.field)}**: {iss.message}")
        else:
            st.success("All checks passed.", icon="✅")

        if result.rules_applied:
            st.markdown("")
            st.markdown("**Memory Applied**")
            for r in result.rules_applied:
                before = r.get("before") or "(empty)"
                st.markdown(f'<span class="rule-pill">{r["field"]}</span> '
                            f'`{before}` → `{r["after"]}`', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3 — Confirm, Teach & Export
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown('<hr class="sd-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-lbl">Step 3 — Confirm, Teach &amp; Export</div>',
                unsafe_allow_html=True)

    if not st.session_state.confirmed:
        col_save, col_ok = st.columns(2)
        with col_save:
            if st.button("💾  Save Corrections & Teach SmartDoc",
                         type="primary", use_container_width=True):
                log, dist = _get_memory()
                doc_id  = st.session_state.invoice_name or \
                          f"inv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                changed = 0
                for fld in TARGET_FIELDS:
                    orig = result.fields.get(fld, "")
                    corr = st.session_state.edits.get(fld, "")
                    if orig != corr:
                        log.record(doc_id=doc_id, field=fld, original=orig,
                                   corrected=corr, template_id="demo")
                        changed += 1
                if changed:
                    dist.fit()
                    st.session_state.confirmed = True
                    st.success(f"✓ {changed} preference(s) saved. "
                               f"SmartDoc now has **{len(dist.rules)} rule(s)** in memory.",
                               icon="🧠")
                    st.balloons()
                else:
                    st.session_state.confirmed = True
                    st.success("No changes — accepted as extracted.", icon="✅")
                st.rerun()
        with col_ok:
            if st.button("✓  Looks Good — No Corrections", use_container_width=True):
                st.session_state.confirmed = True
                st.success("Invoice confirmed with no corrections.", icon="✅")
                st.rerun()

    else:
        st.success("✓ Invoice confirmed. Download your report or process another.", icon="📋")

        if st.session_state.pdf_bytes is None:
            with st.spinner("Generating export PDF…"):
                try:
                    st.session_state.pdf_bytes = build_export_pdf(
                        fields=st.session_state.edits,
                        confidence=result.confidence,
                        validation=result.validation,
                        rules_applied=result.rules_applied,
                        invoice_name=st.session_state.invoice_name,
                        model=result.model,
                    )
                except Exception as e:
                    st.error(f"PDF export failed: {e}", icon="🚨")

        col_dl, col_new = st.columns(2)
        with col_dl:
            if st.session_state.pdf_bytes:
                stem  = Path(st.session_state.invoice_name).stem or "invoice"
                fname = f"SmartDoc_{stem}_{datetime.now().strftime('%Y%m%d')}.pdf"
                st.download_button(
                    label="⬇️  Download Extraction Report (PDF)",
                    data=st.session_state.pdf_bytes,
                    file_name=fname, mime="application/pdf",
                    use_container_width=True,
                )
        with col_new:
            if st.button("📄  Process Another Invoice", use_container_width=True):
                for k, v in {"result": None, "edits": {}, "confirmed": False,
                             "pdf_bytes": None, "invoice_name": ""}.items():
                    st.session_state[k] = v
                st.rerun()

    # Metrics bar
    st.markdown('<hr class="sd-divider">', unsafe_allow_html=True)
    log, dist = _get_memory()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Invoices This Session", st.session_state.sess_processed)
    m2.metric("No-Touch Rate",
              f"{100*st.session_state.sess_notouch//max(st.session_state.sess_processed,1)}%"
              if st.session_state.sess_processed else "—")
    m3.metric("Memory Rules",       len(dist.rules))
    m4.metric("Total Corrections",  log.stats()["total"])
