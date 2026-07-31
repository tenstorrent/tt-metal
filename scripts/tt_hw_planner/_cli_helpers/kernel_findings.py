"""Read + format helpers for ``<demo_dir>/kernel_findings.json``.

The file is produced by ``bringup_plan.collect_bringup_plan_files`` at
scaffold time. It contains the WARN+BLOCKER kernel-constraint findings
that ``kernel_constraints.evaluate_kernels`` computed from the model's
HF config (e.g. "rotary_embedding_hf requires head_dim % 64 == 0; if
you've explicitly set use_hf_rope=True, unset it").

These findings are written ONCE by the planner and then read by two
LLM-facing surfaces:

  * ``_cli_helpers.iter_prompt.build_constraint_block`` — every iter
    prompt gets the findings appended under a model-wide header so the
    LLM doesn't reinvent forbidden patches.
  * ``cli._final_outcome_banner`` — the end-of-run banner surfaces
    findings as "risks" so multi-hour runs don't lose the connection
    between an early static-analysis warning and a late failure.

All functions degrade to empty on missing/corrupt files. The kernel
constraint layer is informational — never block the loop on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_kernel_findings(demo_dir: Path) -> List[Dict[str, Any]]:
    """Read ``<demo_dir>/kernel_findings.json``. Returns the ``findings``
    list (each entry is a dict with op/field/value/constraint/passes/
    severity/fix/source). Returns ``[]`` on any failure — missing file,
    malformed JSON, schema mismatch."""
    if demo_dir is None:
        return []
    path = Path(demo_dir) / "kernel_findings.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    findings = data.get("findings") or []
    if not isinstance(findings, list):
        return []
    out: List[Dict[str, Any]] = []
    for f in findings:
        if isinstance(f, dict):
            out.append(f)
    return out


def _severity_glyph(sev: str) -> str:
    s = (sev or "").lower()
    if s == "blocker":
        return "[BLOCKER]"
    if s == "warn":
        return "[warn]"
    if s == "info":
        return "[info]"
    return "[?]"


def format_kernel_findings_for_prompt(findings: List[Dict[str, Any]]) -> str:
    """Format kernel findings as a markdown block appended to the iter
    prompt. Returns "" if no findings.

    Each finding renders as:
        [warn] <op>.<field>=<value>: <constraint>
              fix: <fix text>

    The block is wrapped in a hard "DO NOT VIOLATE" header because the
    Phi-3.5 attention case proved the LLM will reason its way past
    softer hints (the iter-5 agent explicitly wrote `use_hf_rope=True`
    for "rotary correctness" despite the static analysis flagging it).
    """
    if not findings:
        return ""
    sep = "─" * 72
    lines: List[str] = [
        "",
        sep,
        "MODEL-WIDE KERNEL CONSTRAINTS (from static analysis, do NOT violate)",
        sep,
        "These were derived from the HF config BEFORE the LLM iter loop",
        "started. If your patch would set a config flag that violates one",
        "of these constraints, you will hit a TT_FATAL at runtime — see",
        "the `fix:` line for the workaround.",
        "",
    ]
    for f in findings:
        sev = str(f.get("severity") or "warn")
        op = str(f.get("op") or "?")
        field = str(f.get("field") or "?")
        value = f.get("value")
        constraint = str(f.get("constraint") or "")
        fix = str(f.get("fix") or "")
        glyph = _severity_glyph(sev)
        lines.append(f"  {glyph} {op}.{field}={value}: {constraint}")
        if fix:
            lines.append(f"        fix: {fix}")
    lines.append(sep)
    lines.append("")
    return "\n".join(lines)


def format_kernel_findings_for_banner(findings: List[Dict[str, Any]]) -> List[str]:
    """Format findings as a list of one-line strings suitable for the
    ``extra`` argument of ``_final_outcome_banner``. Each line is
    self-contained (severity glyph + op + constraint + fix in one line)
    because the banner has no multi-line indent convention.

    Returns ``[]`` if no findings.
    """
    if not findings:
        return []
    out: List[str] = []
    out.append("Static-analysis kernel constraints flagged at scaffold time:")
    for f in findings:
        sev = str(f.get("severity") or "warn")
        op = str(f.get("op") or "?")
        field = str(f.get("field") or "?")
        value = f.get("value")
        constraint = str(f.get("constraint") or "")
        fix = str(f.get("fix") or "")
        glyph = _severity_glyph(sev)
        head = f"  {glyph} {op}.{field}={value}: {constraint}"
        out.append(head)
        if fix:
            out.append(f"    fix: {fix}")
    return out
