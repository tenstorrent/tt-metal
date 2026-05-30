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
    from .overlay_manager import load_no_emit_tests, load_persistent_skips

    report_path = demo_dir / "RUN_REPORT.md"

    cat_report = build_final_categorization(model_id=model_id, demo_dir=demo_dir)
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
    lines.append(f"- **ON_DEVICE** ({len(cat_report.on_device)}): graduated, native ttnn, PCC verified")
    if cat_report.on_device:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.on_device))}")
    lines.append(f"- **KERNEL_MISSING** ({len(cat_report.kernel_missing)}): on CPU temporarily — TTNN op gap")
    if cat_report.kernel_missing:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.kernel_missing))}")
    lines.append(f"- **PENDING** ({len(cat_report.pending)}): retry next run")
    if cat_report.pending:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.pending))}")
    lines.append(f"- **structural** ({len(cat_report.structural_excluded)}): " "ModuleList — tested via parent")
    if cat_report.structural_excluded:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.structural_excluded))}")
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
