"""Final placement model.

The torch reference design says every submodule runs on the same device
(GPU). The TT port mirrors that intent: every component goes on the TT
chip, EXCEPT the small residue blocked by a missing TTNN kernel.

Buckets:

  ON_DEVICE       — graduated to native ttnn, PCC verified           → on TT device
  KERNEL_MISSING  — skip-list entry with verified missing TTNN op    → on CPU (temporary)
  PENDING         — not yet graduated, no kernel-missing verdict     → retry next run
                    (the loop hit a tooling-side stop condition like
                    iteration budget or agent-stuck — NOT a placement
                    decision. The component remains on the queue.)

No HOT/COLD: workload firing no longer gates placement. Components that
never fire in a given workload are still device-targets — they just
happen to be inert paths in that workload.

Structural exclusions (ModuleList containers etc.) are listed separately
and not counted toward placement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


_ON_DEVICE = "ON_DEVICE"
_KERNEL_MISSING = "KERNEL_MISSING"
_PENDING = "PENDING"


@dataclass
class CategorizationReport:
    """Structured output: every NEW non-structural component in one of
    three buckets — ON_DEVICE / KERNEL_MISSING / PENDING — plus a
    separate list of structural exclusions."""

    on_device: List[str] = field(default_factory=list)
    kernel_missing: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    structural_excluded: List[str] = field(default_factory=list)

    @property
    def total_categorized(self) -> int:
        return len(self.on_device) + len(self.kernel_missing) + len(self.pending)

    def runtime_target(self, comp: str) -> Optional[str]:
        """Return the runtime placement for ``comp``:
        ``'device'`` for ON_DEVICE, ``'cpu'`` for KERNEL_MISSING (temporary
        fallback while the kernel gap is open), ``None`` for PENDING and
        uncategorized (the loop hasn't decided yet)."""
        if comp in self.on_device:
            return "device"
        if comp in self.kernel_missing:
            return "cpu"
        return None

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            _ON_DEVICE: sorted(self.on_device),
            _KERNEL_MISSING: sorted(self.kernel_missing),
            _PENDING: sorted(self.pending),
        }


def build_final_categorization(
    *,
    model_id: str,
    demo_dir: Path,
    graduated_set: Optional[Set[str]] = None,
) -> CategorizationReport:
    """Walk bringup_status.json + persistent state and bucket every NEW
    non-structural component:

      1. graduated stub on disk → ON_DEVICE
      2. on no-emit list → structural_excluded (NOT placement-categorized)
      3. on skip-list with category == KERNEL_MISSING → KERNEL_MISSING
      4. anything else → PENDING (retry next run; not a permanent state)
    """
    from .overlay_manager import (
        load_no_emit_tests,
        load_persistent_skips,
    )

    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return CategorizationReport()
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        return CategorizationReport()

    new_components = [c.get("name") for c in status.get("components", []) if c.get("status") == "NEW" and c.get("name")]
    if not new_components:
        return CategorizationReport()

    no_emit = set(load_no_emit_tests(model_id).keys())
    skipped = load_persistent_skips(model_id)
    graduated_set = set(graduated_set or _infer_graduated_from_disk(demo_dir, new_components))
    graduated_set -= set(skipped.keys())

    report = CategorizationReport()
    for comp in new_components:
        if comp in no_emit:
            report.structural_excluded.append(comp)
            continue
        if comp in graduated_set:
            report.on_device.append(comp)
            continue
        if comp in skipped and (skipped[comp].get("category") or "").upper() == _KERNEL_MISSING:
            report.kernel_missing.append(comp)
            continue
        # Anything else is in-progress / queue for the next session.
        report.pending.append(comp)
    return report


def _infer_graduated_from_disk(demo_dir: Path, components: List[str]) -> List[str]:
    from .bringup_loop import _safe_id, _stub_has_graduated_from_autofill

    out: List[str] = []
    for comp in components:
        stub = demo_dir / "_stubs" / f"{_safe_id(comp)}.py"
        try:
            if _stub_has_graduated_from_autofill(stub):
                out.append(comp)
        except Exception:
            continue
    return out


def format_categorization_summary(report: CategorizationReport) -> str:
    """Compact placement summary for the demo header / convergence banner."""
    lines: List[str] = []
    lines.append(
        f"  ON_DEVICE      ({len(report.on_device):2}) on TT device:        "
        f"{', '.join(sorted(report.on_device)) or '(none)'}"
    )
    lines.append(
        f"  KERNEL_MISSING ({len(report.kernel_missing):2}) on CPU (TTNN gap):    "
        f"{', '.join(sorted(report.kernel_missing)) or '(none)'}"
    )
    lines.append(
        f"  PENDING        ({len(report.pending):2}) retry next run:       "
        f"{', '.join(sorted(report.pending)) or '(none)'}"
    )
    if report.structural_excluded:
        lines.append(
            f"  (structural — tested via parent, NOT categorized: " f"{', '.join(sorted(report.structural_excluded))})"
        )
    return "\n".join(lines)


def format_kernel_gap_report(model_id: str, report: CategorizationReport) -> str:
    """Surface the KERNEL_MISSING bucket with per-component annotations
    so the TTNN dev planner can prioritize kernel work.

    Only KERNEL_MISSING entries are persisted in the skip-list now, so
    every line in this report represents a verified TTNN op gap.
    """
    if not report.kernel_missing:
        return ""
    from .overlay_manager import load_persistent_skips

    skips = load_persistent_skips(model_id)
    entries: List[str] = []
    for comp in sorted(report.kernel_missing):
        entry = skips.get(comp, {})
        reason = entry.get("reason", "(no annotation)")
        entries.append(f"    {comp:40} → {reason}")
    if not entries:
        return ""
    lines: List[str] = ["", f"  TTNN OPERATIONS NEEDED ({len(entries)}):"]
    lines.extend(entries)
    lines.append("")
    lines.append("  These components stay on CPU until TTNN lands the missing op(s).")
    lines.append("")
    return "\n".join(lines)
