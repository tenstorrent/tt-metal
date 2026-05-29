"""Final 3-category placement model.

Every NEW component lands in exactly ONE of three categories. Each
category has intrinsic runtime placement:

  HOT            — invoked in workload, graduated to native ttnn → on TT device
  COLD           — not invoked in workload                       → on CPU
  KERNEL_MISSING — invoked, verified TTNN op missing             → on CPU

No DROPPED bucket: ModuleList containers are STRUCTURAL EXCLUSIONS
(filtered out before categorization, mentioned in the report as a side
note, not assigned a category).

No HOT_STUCK / UNCLASSIFIED: the tool's invariant is "every HOT
component graduates OR is verified-kernel-missing." A component that
neither graduates nor verifies kernel-missing means the tool isn't
finished — surface as run-level TOOL FAILURE, not as a category.

Placement is INTRINSIC to the category:
  category == HOT             → runtime_target == "device"
  category == COLD            → runtime_target == "cpu"
  category == KERNEL_MISSING  → runtime_target == "cpu"

The skip-list entry's `category` field carries the COLD vs
KERNEL_MISSING distinction. A graduated component is not on the
skip-list; that absence + its native ttnn stub is the HOT signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


# The three valid categories. No DROPPED, no HOT_STUCK, no UNCLASSIFIED.
_HOT = "HOT"
_COLD = "COLD"
_KERNEL_MISSING = "KERNEL_MISSING"


@dataclass
class CategorizationReport:
    """Structured output: every NEW non-structural component in exactly one
    of three categories, plus a separate list of structural exclusions."""

    hot: List[str] = field(default_factory=list)
    cold: List[str] = field(default_factory=list)
    kernel_missing: List[str] = field(default_factory=list)
    # Structural exclusions (ModuleList containers etc.). NOT a category.
    # Listed for the demo header but not part of the placement decision.
    structural_excluded: List[str] = field(default_factory=list)

    @property
    def total_categorized(self) -> int:
        return len(self.hot) + len(self.cold) + len(self.kernel_missing)

    def runtime_target(self, comp: str) -> Optional[str]:
        """Return the intrinsic placement for ``comp``: 'device' iff HOT,
        'cpu' iff COLD or KERNEL_MISSING, None if comp isn't categorized."""
        if comp in self.hot:
            return "device"
        if comp in self.cold or comp in self.kernel_missing:
            return "cpu"
        return None

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            _HOT: sorted(self.hot),
            _COLD: sorted(self.cold),
            _KERNEL_MISSING: sorted(self.kernel_missing),
        }


def build_final_categorization(
    *,
    model_id: str,
    demo_dir: Path,
    graduated_set: Optional[Set[str]] = None,
) -> CategorizationReport:
    """Walk bringup_status.json + persistent state and assign every
    NEW non-structural component to exactly one of HOT/COLD/KERNEL_MISSING.

    Routing:
      1. graduated → HOT (on TT device, PCC verified)
      2. on no-emit list → structural_excluded (NOT categorized)
      3. on skip-list with category == "COLD" → COLD
      4. on skip-list with category == "KERNEL_MISSING" → KERNEL_MISSING
      5. on skip-list with no/other category → COLD (default safe choice
         until classify-hot-cold OR kernel-missing verification refines it)
    """
    from .overlay_manager import load_hot_cold, load_no_emit_tests, load_persistent_skips

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
    skipped = load_persistent_skips(model_id)  # dict {comp: {category, reason, ...}}
    hot_cold_map = load_hot_cold(model_id)  # dict {comp: "HOT" | "COLD"}
    graduated_set = set(graduated_set or _infer_graduated_from_disk(demo_dir, new_components))
    # A skip-listed component CANNOT be PCC-graduated.
    graduated_set -= set(skipped.keys())

    report = CategorizationReport()
    for comp in new_components:
        if comp in no_emit:
            # Structural exclusion (ModuleList container). NOT a category.
            report.structural_excluded.append(comp)
            continue
        if comp in graduated_set:
            report.hot.append(comp)
            continue
        if comp in skipped:
            cat = skipped[comp].get("category", "").upper()
            # Cross-check: the workload probe might have classified this
            # as HOT after the skip-list entry was written. If hot_cold
            # says HOT and skip-list says COLD, the workload signal wins
            # (more reliable). Promote to KERNEL_MISSING since a HOT
            # component on CPU means TTNN dev work or tool budget issue.
            hc_kind = hot_cold_map.get(comp, "").upper()
            # KERNEL_MISSING and CONSTRAINT_MISMATCH both map to the
            # KERNEL_MISSING placement bucket: in both cases the TTNN
            # stack needs work (a new op, or new dtype/layout support
            # on an existing op) before the component can move off CPU.
            # TOOL_BUG / HF_ERROR / ITERATION_BUDGET / AGENT_STUCK map
            # to COLD placement (they may or may not be permanent;
            # next-run retry is governed by the detailed category in
            # the skip-list, not the 3-bucket placement model).
            if cat in (_KERNEL_MISSING, "CONSTRAINT_MISMATCH"):
                report.kernel_missing.append(comp)
            elif cat == _COLD and hc_kind == _HOT:
                report.kernel_missing.append(comp)
            else:
                report.cold.append(comp)
            continue
        # Not graduated, not on any list.
        # Consult hot_cold.json (workload probe result):
        #   - "COLD"  → goes to COLD (CPU is correct)
        #   - "HOT"   → tool didn't finish processing this HOT comp.
        #               Route to KERNEL_MISSING with implicit "tool
        #               didn't reach it" annotation.
        #   - missing → default COLD (safest CPU placement)
        hc_kind = hot_cold_map.get(comp, "").upper()
        if hc_kind == _HOT:
            report.kernel_missing.append(comp)
        else:
            report.cold.append(comp)
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
    """Compact 3-category summary for the demo header / convergence banner."""
    lines: List[str] = []
    lines.append(
        f"  HOT            ({len(report.hot):2}) on TT device:       {', '.join(sorted(report.hot)) or '(none)'}"
    )
    lines.append(
        f"  COLD           ({len(report.cold):2}) on CPU (intended):  {', '.join(sorted(report.cold)) or '(none)'}"
    )
    lines.append(
        f"  KERNEL_MISSING ({len(report.kernel_missing):2}) on CPU (TTNN gap):  {', '.join(sorted(report.kernel_missing)) or '(none)'}"
    )
    if report.structural_excluded:
        lines.append(
            f"  (structural — tested via parent, NOT categorized: " f"{', '.join(sorted(report.structural_excluded))})"
        )
    return "\n".join(lines)


def format_kernel_gap_report(model_id: str, report: CategorizationReport) -> str:
    """Surface the KERNEL_MISSING placement bucket with per-component
    annotations so the TTNN dev planner can prioritize work.

    Splits the bucket into two sections based on the underlying detailed
    skip-list category:

      * Missing-op section   — KERNEL_MISSING entries (TTNN truly lacks
                                the op). TTNN dev work: implement the op.
      * Constraint section   — CONSTRAINT_MISMATCH entries (TTNN has the
                                op but failed for the call's dtype /
                                layout / shape). TTNN dev work: extend
                                the existing op's support matrix.
    """
    if not report.kernel_missing:
        return ""
    from .overlay_manager import load_persistent_skips

    skips = load_persistent_skips(model_id)
    missing_op_entries: List[str] = []
    constraint_entries: List[str] = []
    other_entries: List[str] = []
    for comp in sorted(report.kernel_missing):
        entry = skips.get(comp, {})
        reason = entry.get("reason", "(no annotation)")
        cat = (entry.get("category") or "").upper()
        line = f"    {comp:40} → {reason}"
        if cat == _KERNEL_MISSING:
            missing_op_entries.append(line)
        elif cat == "CONSTRAINT_MISMATCH":
            constraint_entries.append(line)
        else:
            # E.g. hot_cold-HOT-but-skip-COLD promotion (no specific
            # missing-op or constraint signal). Surface separately.
            other_entries.append(line)

    lines: List[str] = [""]
    if missing_op_entries:
        lines.append(f"  TTNN OPERATIONS NEEDED ({len(missing_op_entries)}):")
        lines.extend(missing_op_entries)
        lines.append("")
        lines.append("  These components stay on CPU until TTNN lands the missing op(s).")
        lines.append("")
    if constraint_entries:
        lines.append(f"  TTNN CONSTRAINT EXTENSIONS NEEDED ({len(constraint_entries)}):")
        lines.extend(constraint_entries)
        lines.append("")
        lines.append(
            "  These components stay on CPU until TTNN extends the existing op's " "dtype/layout/shape support."
        )
        lines.append("")
    if other_entries:
        lines.append(f"  HOT-PATH ON CPU — TTNN GAP UNCLASSIFIED ({len(other_entries)}):")
        lines.extend(other_entries)
        lines.append("")
        lines.append(
            "  Workload-HOT but the tool couldn't pin a specific missing op or "
            "constraint signal. Investigation needed."
        )
        lines.append("")
    return "\n".join(lines)
