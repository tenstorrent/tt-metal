"""Markdown rendering for issue-verifier results.

Output goes to the Actions job summary, so it is read by someone deciding
whether to put a bounty on an issue. Lead with the verdict and show the numbers
that produced it; a reader should be able to disagree with the tool.
"""

from __future__ import annotations

import math
from typing import Any

from issue_verifier.verdict import Assessment, Verdict, is_categorical

BADGES = {
    Verdict.CLAIM_UNSOUND: "❌ CLAIM UNSOUND",
    Verdict.NOT_REPRODUCED: "⚪ NOT REPRODUCED",
    Verdict.LIKELY_VALID: "✅ LIKELY VALID",
    Verdict.NEEDS_HARDWARE: "🔶 NEEDS HARDWARE",
    Verdict.NEEDS_HUMAN: "🔷 NEEDS HUMAN",
}


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return f"`{value}`"
    if isinstance(value, float):
        if math.isnan(value):
            return "`nan`"
        if math.isinf(value):
            return "`inf`" if value > 0 else "`-inf`"
        return f"`{value:.6g}`"
    text = str(value)
    return f"`{text[:60]}`" if text else "`—`"


def _rows_table(rows: list[dict], left: str, right: str, *, grade: bool = False) -> str:
    if not rows:
        return "_No rows recorded._\n"

    header = f"| case | {left} | {right} | agree |"
    divider = "|---|---|---|---|"
    if grade:
        header += " kind |"
        divider += "---|"

    out = [header, divider]
    for row in rows:
        agrees = bool(row.get("agree"))
        line = f"| {row.get('name', '?')} | {_fmt(row.get(left))} | {_fmt(row.get(right))} | "
        line += "yes |" if agrees else "**no** |"
        if grade:
            if agrees:
                kind = "—"
            else:
                kind = "**categorical**" if is_categorical(row) else "marginal"
            line += f" {kind} |"
        out.append(line)
    return "\n".join(out) + "\n"


def _gate_header(name: str, gate: dict, title: str) -> str:
    if gate.get("ran"):
        return f"### {name} — {title}\n\n"
    reason = gate.get("skipped_reason") or gate.get("error") or "not run"
    return f"### {name} — {title}\n\n_Skipped: {reason}_\n\n"


def render(
    *,
    issue: dict,
    plan: dict,
    measurements: dict,
    assessment: Assessment,
    sku: str,
) -> str:
    number = issue.get("number")
    parts: list[str] = []

    parts.append(f"## {BADGES[assessment.verdict]} — issue #{number}\n")
    parts.append(f"**{assessment.headline}**\n")
    parts.append(f"> [{issue.get('title', '')}](https://github.com/tenstorrent/tt-metal/issues/{number})\n")

    for reason in assessment.reasons:
        parts.append(f"- {reason}\n")
    parts.append("\n")

    if assessment.flags:
        parts.append("**Worth knowing before you act on this:**\n\n")
        for flag in assessment.flags:
            parts.append(f"- {flag}\n")
        parts.append("\n")

    if not plan.get("verifiable", False):
        parts.append("---\n\nNo experiment was run, so there is nothing further to show.\n")
        return "".join(parts)

    parts.append("---\n\n")
    parts.append(f"**Claim.** {plan.get('claim_summary', '—')}\n\n")
    parts.append(f"**Op.** `{plan.get('op', '—')}` · **type** `{plan.get('claim_type', '—')}` · ")
    parts.append(f"**ran on** `{sku}`\n\n")
    if plan.get("sku_rationale"):
        parts.append(f"_Hardware choice: {plan['sku_rationale']}_\n\n")

    gate_a = measurements.get("gate_a") or {}
    parts.append(_gate_header("Gate A", gate_a, "the report's expected values vs. the real reference"))
    if gate_a.get("ran"):
        parts.append(_rows_table(gate_a.get("rows") or [], "claimed_expected", "reference", grade=True))
        parts.append(
            "\n_`categorical` means the reported value could not have come from the reference "
            "(finite where it returns infinity, or wrong by more than 0.1%). `marginal` means it "
            "is close enough to be a rounded printout and is not held against the report._\n"
        )
        parts.append("\n")

    gate_b = measurements.get("gate_b") or {}
    parts.append(_gate_header("Gate B", gate_b, "was this behavior already settled deliberately?"))
    findings = [f for f in (gate_b.get("findings") or []) if f.get("commit")]
    if gate_b.get("ran"):
        if findings:
            parts.append("| commit | subject | on purpose? | relevance |\n|---|---|---|---|\n")
            for f in findings:
                intent = "**yes**" if f.get("deliberate") else "no"
                parts.append(
                    f"| `{f.get('commit')}` | {f.get('subject', '?')} | {intent} | {f.get('why_relevant', '')} |\n"
                )
        else:
            parts.append("_No commit in the history of the cited files touches this behavior._\n")
        parts.append("\n")

    gate_c = measurements.get("gate_c") or {}
    parts.append(_gate_header("Gate C", gate_c, "the op on real silicon vs. the reference"))
    if gate_c.get("ran"):
        parts.append(_rows_table(gate_c.get("rows") or [], "device", "reference"))
        parts.append("\n")

    gate_d = measurements.get("gate_d") or {}
    parts.append(_gate_header("Gate D", gate_d, "would the proposed change hold up?"))
    if gate_d.get("ran"):
        parts.append("Cases from the report:\n\n")
        parts.append(_rows_table(gate_d.get("rows") or [], "counterfactual", "reference"))
        parts.append("\nMirror cases stressing the opposite extreme:\n\n")
        parts.append(_rows_table(gate_d.get("mirror_rows") or [], "counterfactual", "reference"))
        if gate_d.get("mirror_rationale"):
            parts.append(f"\n_{gate_d['mirror_rationale']}_\n")
        parts.append("\n")

    if measurements.get("notes"):
        parts.append(f"**Notes.** {measurements['notes']}\n\n")

    parts.append("---\n")
    parts.append(
        "Generated by `.github/issue_verifier`. Every number above was printed by a script "
        "run during this job. The verdict is a fixed rule over those numbers, not a model's "
        "opinion — but the experiment design was model-generated, so a surprising result "
        "deserves a look at the probe before it is quoted.\n"
    )
    return "".join(parts)
