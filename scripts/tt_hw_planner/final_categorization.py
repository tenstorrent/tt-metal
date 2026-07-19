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

A decomposed parent earns ON_DEVICE only by passing its OWN test. While
it is split (no_emit) it is PENDING — children all graduating does NOT
credit the parent. The recompose path restores the parent as a whole-
module target once its children are on device; when that recomposed
module passes PCC its stub graduates and it lands ON_DEVICE the same as
any other component. A no_emit parent blocked by a kernel-missing child
rolls up as KERNEL_MISSING. There is no separate "structural" bucket —
every NEW component lands in exactly one of the three placement buckets.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


_ON_DEVICE = "ON_DEVICE"
_KERNEL_MISSING = "KERNEL_MISSING"
_PENDING = "PENDING"
_CPU_REUSE = "CPU_REUSE"


@dataclass
class CategorizationReport:
    """Structured output: every component lands in exactly one bucket.

    ON_DEVICE      — NEW graduated to native ttnn (PCC + native-body verified),
                     or a REUSE/ADAPT whose target is genuinely wired into this
                     demo's device tree.
    KERNEL_MISSING — verified missing TTNN op → on CPU (temporary).
    PENDING        — NEW not yet graduated, no kernel-missing verdict.
    CPU_REUSE      — a REUSE/ADAPT tag that is NOT verified on device (no
                     graduated stub AND its reuse target is not wired into the
                     demo). It runs on CPU via the eager runner. This bucket
                     exists so such components are reported honestly instead of
                     being counted on-device on a tag alone, or dropped."""

    on_device: List[str] = field(default_factory=list)
    kernel_missing: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    cpu_reuse: List[str] = field(default_factory=list)

    @property
    def total_categorized(self) -> int:
        return len(self.on_device) + len(self.kernel_missing) + len(self.pending) + len(self.cpu_reuse)

    def runtime_target(self, comp: str) -> Optional[str]:
        """Return the runtime placement for ``comp``:
        ``'device'`` for ON_DEVICE, ``'cpu'`` for KERNEL_MISSING (temporary
        fallback while the kernel gap is open) or CPU_REUSE (unverified reuse
        running on the eager runner), ``None`` for PENDING and uncategorized
        (the loop hasn't decided yet)."""
        if comp in self.on_device:
            return "device"
        if comp in self.kernel_missing or comp in self.cpu_reuse:
            return "cpu"
        return None

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            _ON_DEVICE: sorted(self.on_device),
            _KERNEL_MISSING: sorted(self.kernel_missing),
            _PENDING: sorted(self.pending),
            _CPU_REUSE: sorted(self.cpu_reuse),
        }


def _reuse_target_wired(demo_dir: Path, reuse_target: Optional[str]) -> bool:
    """True when a REUSE/ADAPT component's tt reuse target is genuinely wired
    into THIS demo — i.e. the sibling ttnn module was copied into the demo's own
    device tree (``tt/`` or ``ttnn/``, the template-scaffold signal) OR is
    explicitly imported by a non-stub, non-test demo file.

    A plain REUSE tag is NOT enough: on the generic hf_eager (CPU-eager) path no
    sibling module is copied or imported, so the component actually runs the HF
    reference on CPU. This check is what distinguishes a real on-device reuse
    from a tag that silently falls to torch-on-CPU. Never raises."""
    if not reuse_target:
        return False
    tok = str(reuse_target).strip().split(" ")[0].rstrip("/")
    if not tok.endswith(".py") or "/" not in tok:
        return False
    stem = Path(tok).name
    dotted = tok[:-3].replace("/", ".")
    modname = dotted.rsplit(".", 1)[-1]
    try:
        for sub in ("tt", "ttnn"):
            d = demo_dir / sub
            if d.is_dir():
                for f in d.rglob(stem):
                    if f.is_file():
                        return True
        import_re = re.compile(rf"(?:^|\s)import\s+{re.escape(modname)}\b|from\s+\S*{re.escape(modname)}\s+import")
        for py in demo_dir.rglob("*.py"):
            parts = set(py.parts)
            if "_stubs" in parts or "tests" in parts:
                continue
            try:
                txt = py.read_text(errors="replace")
            except Exception:
                continue
            if dotted in txt or import_re.search(txt):
                return True
    except Exception:
        return False
    return False


def reuse_adapt_on_device(demo_dir: Path, components_list: List[dict]) -> Set[str]:
    """Set of REUSE/ADAPT component names that are VERIFIED on device — either a
    graduated native stub exists (rare for REUSE) or the reuse target is wired
    (:func:`_reuse_target_wired`). Shared by both the terminal compute-split and
    the RUN_REPORT categorization so the two surfaces cannot disagree."""
    from .bringup_loop import _safe_id, _stub_has_graduated_any

    out: Set[str] = set()
    for c in components_list:
        name = c.get("name")
        if not name or c.get("status") not in ("REUSE", "ADAPT"):
            continue
        stub = demo_dir / "_stubs" / f"{_safe_id(name)}.py"
        try:
            graduated = _stub_has_graduated_any(stub)
        except Exception:
            graduated = False
        if graduated or _reuse_target_wired(demo_dir, c.get("tt_reuse_target")):
            out.add(name)
    return out


def build_final_categorization(
    *,
    model_id: str,
    demo_dir: Path,
    graduated_set: Optional[Set[str]] = None,
) -> CategorizationReport:
    """Walk bringup_status.json + persistent state and bucket every NEW
    component:

      1. graduated stub on disk → ON_DEVICE (parents earn this by passing
         their own recomposed test, same as any component)
      2. decomposed parent (no-emit) → KERNEL_MISSING if a child is blocked
         by a missing TTNN op, else PENDING (children graduating does not
         credit the parent; recompose must prove the whole module)
      3. on skip-list with category == KERNEL_MISSING → KERNEL_MISSING
      4. anything else → PENDING (retry next run; not a permanent state)
    """
    from .overlay_manager import (
        load_alias_credits,
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

    components_list = status.get("components", []) or []
    new_components = [c.get("name") for c in components_list if c.get("status") == "NEW" and c.get("name")]
    reuse_adapt = [c.get("name") for c in components_list if c.get("status") in ("REUSE", "ADAPT") and c.get("name")]
    if not new_components and not reuse_adapt:
        return CategorizationReport()

    sp_of: Dict[str, str] = {c["name"]: (c.get("submodule_path") or "") for c in components_list if c.get("name")}
    sp_to_comp: Dict[str, str] = {sp: n for n, sp in sp_of.items() if sp}

    no_emit = set(load_no_emit_tests(model_id).keys())
    skipped = load_persistent_skips(model_id)
    if graduated_set is None:
        graduated_set = set(_infer_graduated_from_disk(demo_dir, new_components))
    else:
        graduated_set = set(graduated_set)
    graduated_set -= set(skipped.keys())
    graduated_set |= set(load_alias_credits(model_id).keys())

    kernel_missing_set = {n for n, e in skipped.items() if (e.get("category") or "").upper() == _KERNEL_MISSING}
    decomp_children = _load_decomposition_children(demo_dir)

    report = CategorizationReport()
    for comp in new_components:
        if comp in graduated_set:
            report.on_device.append(comp)
            continue
        if comp in no_emit:
            if _blocked_by_kernel(comp, decomp_children, sp_to_comp, kernel_missing_set, frozenset()):
                report.kernel_missing.append(comp)
            else:
                report.pending.append(comp)
            continue
        if comp in kernel_missing_set:
            report.kernel_missing.append(comp)
            continue
        report.pending.append(comp)

    verified_reuse = reuse_adapt_on_device(demo_dir, components_list)
    for comp in reuse_adapt:
        if comp in verified_reuse:
            report.on_device.append(comp)
        else:
            report.cpu_reuse.append(comp)
    return report


def _blocked_by_kernel(
    name: str,
    decomp_children: Dict[str, List[str]],
    sp_to_comp: Dict[str, str],
    kernel_missing_set: Set[str],
    stack: frozenset,
) -> bool:
    if name in stack:
        return False
    kids = decomp_children.get(name)
    if not kids:
        return False
    nxt = stack | {name}
    for sp in kids:
        comp = sp_to_comp.get(sp)
        if comp is None:
            continue
        if comp in kernel_missing_set:
            return True
        if _blocked_by_kernel(comp, decomp_children, sp_to_comp, kernel_missing_set, nxt):
            return True
    return False


def parents_ready_to_recompose(
    *,
    model_id: str,
    demo_dir: Path,
    graduated_set: Optional[Set[str]] = None,
) -> List[str]:
    """Return the decomposed (no_emit) parents whose every decomposition
    child is on device — ready to be restored as a whole-module target and
    recomposed.

    A child counts as on device if its stub graduated, or it is itself a
    decomposed parent whose children are all on device (recursive). A parent
    with no recorded children, or with any child not yet on device, is NOT
    ready (it stays PENDING).
    """
    from .overlay_manager import load_no_emit_tests, load_persistent_skips

    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return []
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        return []
    components_list = status.get("components", []) or []
    new_components = [c.get("name") for c in components_list if c.get("status") == "NEW" and c.get("name")]
    if not new_components:
        return []

    sp_to_comp = {
        (c.get("submodule_path") or ""): c.get("name")
        for c in components_list
        if c.get("name") and c.get("submodule_path")
    }
    reuse_set = {c.get("name") for c in components_list if c.get("status") == "REUSE" and c.get("name")}
    no_emit = set(load_no_emit_tests(model_id).keys())
    skipped = load_persistent_skips(model_id)
    graduated_set = set(graduated_set or set()) | set(_infer_graduated_from_disk(demo_dir, new_components))
    graduated_set -= set(skipped.keys())
    decomp_children = _load_decomposition_children(demo_dir)
    memo: Dict[str, bool] = {}

    def _covered(name: str, stack: frozenset) -> bool:
        if name in memo:
            return memo[name]
        if name in stack:
            return False
        if name in graduated_set:
            memo[name] = True
            return True
        kids = decomp_children.get(name)
        if not kids:
            memo[name] = False
            return False
        nxt = stack | {name}
        ok = all(_sp_cov(sp, nxt) for sp in kids)
        memo[name] = ok
        return ok

    def _sp_cov(submodule_path: str, stack: frozenset) -> bool:
        comp = sp_to_comp.get(submodule_path)
        if comp is None:
            return False
        if comp in graduated_set:
            return True
        if comp in reuse_set:
            return True
        if comp in no_emit:
            return _covered(comp, stack)
        return False

    ready: List[str] = []
    for parent in sorted(no_emit):
        if parent in graduated_set:
            continue
        if parent not in decomp_children:
            continue
        if _covered(parent, frozenset()):
            ready.append(parent)
    return ready


def _load_decomposition_children(demo_dir: Path) -> Dict[str, List[str]]:
    """Map each decomposed parent → its intended child submodule_paths,
    from the live decomposition_plan.json plus any archived plans."""
    plans: List[Path] = []
    live = demo_dir / "decomposition_plan.json"
    if live.is_file():
        plans.append(live)
    arch = demo_dir / "decomposition_plan.applied"
    if arch.is_dir():
        plans.extend(sorted(arch.glob("plan_*.json")))

    out: Dict[str, List[str]] = {}
    for p in plans:
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            parent = entry.get("parent_name")
            kids = entry.get("children") or []
            paths = [c.get("submodule_path") for c in kids if isinstance(c, dict) and c.get("submodule_path")]
            if parent and paths:
                out[parent] = paths
    return out


def _infer_graduated_from_disk(demo_dir: Path, components: List[str]) -> List[str]:
    from .bringup_loop import _safe_id, _stub_has_graduated_any

    out: List[str] = []
    for comp in components:
        stub = demo_dir / "_stubs" / f"{_safe_id(comp)}.py"
        try:
            if _stub_has_graduated_any(stub):
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
    if report.cpu_reuse:
        lines.append(
            f"  CPU_REUSE      ({len(report.cpu_reuse):2}) reuse tag NOT wired:  "
            f"{', '.join(sorted(report.cpu_reuse))}"
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
