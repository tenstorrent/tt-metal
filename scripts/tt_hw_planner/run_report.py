"""Generate ``RUN_REPORT.md`` — a markdown summary of an auto-iterate run.

Written to ``<demo_dir>/RUN_REPORT.md`` after the auto-iterate loop
finishes (converged or not). Captures everything the operator needs
to understand WHAT the tool decided and WHY:

  * 3-category placement summary (HOT / COLD / KERNEL_MISSING)
  * Cold-evidence per component (kind + signals + reasons)
  * Skip-list contents (grouped by category)
  * Graduated stubs list
  * Demo emission status

Pure read-side: this module only READS state files; the auto-iterate
loop calls it once at the end. Failure to write the report never
blocks the run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def emit_run_report(
    model_id: str,
    demo_dir: Path,
    *,
    converged: Optional[bool] = None,
    iterations_run: Optional[int] = None,
    demo_emit_status: Optional[str] = None,
    demo_pytest_status: Optional[str] = None,
) -> Optional[Path]:
    """Write ``<demo_dir>/RUN_REPORT.md`` summarizing the run.

    Returns the path written, or ``None`` if write failed (failure is
    non-fatal — never propagates).
    """
    try:
        return _emit_run_report_impl(
            model_id,
            demo_dir,
            converged=converged,
            iterations_run=iterations_run,
            demo_emit_status=demo_emit_status,
            demo_pytest_status=demo_pytest_status,
        )
    except Exception as exc:
        import sys

        print(f"  [run-report] failed to emit RUN_REPORT.md: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _emit_run_report_impl(
    model_id: str,
    demo_dir: Path,
    *,
    converged: Optional[bool],
    iterations_run: Optional[int],
    demo_emit_status: Optional[str],
    demo_pytest_status: Optional[str],
) -> Path:
    from .final_categorization import build_final_categorization
    from .overlay_manager import load_hot_cold_evidence, load_no_emit_tests, load_persistent_skips

    report_path = demo_dir / "RUN_REPORT.md"

    cat_report = build_final_categorization(model_id=model_id, demo_dir=demo_dir)
    evidence = load_hot_cold_evidence(model_id)
    skips = load_persistent_skips(model_id)
    no_emit = load_no_emit_tests(model_id)

    lines: List[str] = []
    lines.append(f"# Bring-up run report — `{model_id}`")
    lines.append("")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_")
    lines.append("")
    lines.append("## Outcome")
    lines.append("")
    if converged is True:
        lines.append(f"**Converged** after {iterations_run or '?'} iteration(s).")
    elif converged is False:
        lines.append(f"**Did not converge** after {iterations_run or '?'} iteration(s).")
    else:
        lines.append("Run state: report generated outside the auto-iterate loop.")
    if demo_emit_status:
        lines.append(f"- Demo emission: `{demo_emit_status}`")
    if demo_pytest_status:
        lines.append(f"- Demo pytest:   `{demo_pytest_status}`")
    lines.append("")

    lines.append("## Placement summary")
    lines.append("")
    lines.append(f"- **HOT** ({len(cat_report.hot)}): on TT device")
    if cat_report.hot:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.hot))}")
    lines.append(f"- **COLD** ({len(cat_report.cold)}): on CPU — evidence shows no device value")
    if cat_report.cold:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.cold))}")
    lines.append(f"- **KERNEL_MISSING** ({len(cat_report.kernel_missing)}): on CPU — TTNN gap")
    if cat_report.kernel_missing:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.kernel_missing))}")
    lines.append(f"- **structural** ({len(cat_report.structural_excluded)}): " "ModuleList — tested via parent")
    if cat_report.structural_excluded:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.structural_excluded))}")
    lines.append("")

    if evidence:
        lines.append("## Per-component evidence")
        lines.append("")
        lines.append("| Component | Kind | Freq | CPU ms | CPU % | Density | Affinity | Reasons |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name in sorted(evidence.keys()):
            e = evidence[name]
            kind = str(e.get("kind", "?"))
            freq = e.get("frequency")
            lat_ms = e.get("cpu_latency_ms")
            lat_pct = e.get("cpu_latency_pct")
            dens = e.get("compute_density")
            aff = e.get("affinity_score")
            reasons = e.get("evidence", [])
            freq_s = "—" if freq is None else f"{freq:.2f}"
            lat_ms_s = "—" if lat_ms is None else f"{lat_ms:.2f}"
            lat_pct_s = "—" if lat_pct is None else f"{lat_pct:.2f}%"
            dens_s = "—" if (not dens or dens == 0) else f"{dens:.2e}"
            aff_s = "—" if aff is None else f"{aff:+d}"
            why = "; ".join(str(r) for r in reasons).replace("|", "\\|")[:120]
            lines.append(
                f"| `{name}` | **{kind}** | {freq_s} | {lat_ms_s} | {lat_pct_s} | " f"{dens_s} | {aff_s} | {why} |"
            )
        lines.append("")

    if skips:
        # Group by category for clearer reading
        by_cat: Dict[str, List[tuple]] = {}
        for name, entry in skips.items():
            cat = (entry.get("category") or "UNKNOWN").upper()
            by_cat.setdefault(cat, []).append((name, entry))

        lines.append("## Skip-list")
        lines.append("")
        for cat in sorted(by_cat.keys()):
            entries = by_cat[cat]
            lines.append(f"### `{cat}` — {len(entries)} entry")
            lines.append("")
            for name, entry in sorted(entries, key=lambda x: x[0]):
                reason = (entry.get("reason") or "").replace("\n", " ")[:200]
                retries = entry.get("retry_count")
                retry_s = f" (retries: {retries})" if retries else ""
                lines.append(f"- `{name}`{retry_s}: {reason}")
            lines.append("")

    if no_emit:
        lines.append("## Structurally excluded (no_emit)")
        lines.append("")
        for name, entry in sorted(no_emit.items(), key=lambda x: x[0]):
            reason = (entry.get("reason") or "").replace("\n", " ")[:200]
            lines.append(f"- `{name}`: {reason}")
        lines.append("")

    lines.append("## Next steps")
    lines.append("")
    if cat_report.kernel_missing:
        lines.append(
            f"- **{len(cat_report.kernel_missing)} component(s) need TTNN work** — see KERNEL_MISSING section above"
        )
    retryable_count = sum(
        1 for entry in skips.values() if (entry.get("category") or "").upper() in {"ITERATION_BUDGET", "AGENT_STUCK"}
    )
    if retryable_count:
        lines.append(f"- **{retryable_count} retryable skip(s)** — next `up --auto` re-attempts them automatically")
    tool_bugs = sum(1 for entry in skips.values() if (entry.get("category") or "").upper() == "TOOL_BUG")
    if tool_bugs:
        lines.append(
            f"- **{tool_bugs} TOOL_BUG entry** — fix the scaffolder, then run "
            f"`overlay-clear-skips --category TOOL_BUG {model_id}`"
        )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    return report_path
