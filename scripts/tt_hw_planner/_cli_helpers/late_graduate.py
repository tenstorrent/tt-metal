"""Late-graduate path for components discovered during e2e synthesis (Item 4).

When :func:`run_e2e_synthesis_loop` (Item 3) discovers that the chained
forward needs a TTNN module the per-component decomposition didn't
identify, the tool needs a way to graduate that module on the fly:

  1. Add the new component to ``bringup_status.json`` with the
     ``LATE_GRADUATE`` status flag (distinguishes it from the
     decomposer's static-analysis output).
  2. Run the per-component LLM iterate against it — same loop shape
     and same PCC ≥ 0.99 gate as the original components.
  3. On graduation: resume the e2e synthesis loop with the new
     component available.
  4. On graduation failure: route to CPU fallback (same as
     KERNEL_MISSING) so the e2e synthesis can continue without it.

The status flag matters: future runs (or rerun-aware tooling) can see
which components were decomposer-found vs synthesis-discovered, and
the planner registry can record the late-discovery in
``learned_bringups.json`` so the next sibling model starts with the
fuller manifest.

Distinct from the existing per-component iterate loop in
:mod:`_cli_helpers.auto_iterate`:

  * That loop is one-shot: takes the full manifest at startup and
    runs every component through scaffold → autofill → iterate.
  * This module is single-component, callable on demand from any
    other layer (specifically Item 3's synthesis loop).

It REUSES :func:`llm_synth.synthesize_component` for the actual LLM
call — same prompt path, same agent dispatch, same retry logic the
batch loop uses. The new code here is the manifest mutation + status
flag, not the synthesis itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


LATE_GRADUATE_STATUS = "LATE_GRADUATE"


@dataclass
class LateGraduateComponentSpec:
    """Description of a single late-discovered component.

    Carries the minimum information the decomposer would have produced
    if it had found this module during static analysis. The
    discoverer (Item 3's synthesis loop) is responsible for filling
    these from the HF model + its own analysis.
    """

    name: str  # human-readable; used as bringup_status key
    hf_reference: str  # dotted path into HF model, e.g. "vision_encoder.fpn"
    class_name: str  # HF class name, e.g. "ConvFPNAdapter"
    extras: Dict[str, Any] = field(default_factory=dict)  # passthrough manifest fields

    def to_manifest_entry(self) -> Dict[str, Any]:
        """Render as a bringup_status.json component entry."""
        entry = {
            "name": self.name,
            "status": LATE_GRADUATE_STATUS,
            "hf_reference": self.hf_reference,
            "class_name": self.class_name,
        }
        entry.update(self.extras)
        return entry


@dataclass
class LateGraduateResult:
    """Outcome of one late-graduate attempt."""

    component: str
    converged: bool  # True iff per-component PCC ≥ 0.99 was reached
    pcc: Optional[float] = None
    iters_used: int = 0
    fallback_to_cpu: bool = False  # True iff routed to CPU after exhaustion
    diagnostic: str = ""


# ─── Manifest mutation ──────────────────────────────────────────────


def add_late_graduate_to_manifest(
    demo_dir: Path,
    spec: LateGraduateComponentSpec,
) -> bool:
    """Append (or update) ``spec`` in ``demo_dir/bringup_status.json``
    with ``status=LATE_GRADUATE``.

    Idempotent: if a component with the same name already exists in
    the manifest, the existing entry is UPDATED in place (status flips
    to ``LATE_GRADUATE`` if it wasn't already, ``hf_reference`` /
    ``class_name`` overwritten with the new spec). This is the right
    behavior when synthesis re-discovers the same module across multiple
    iters — we don't want duplicate entries.

    Returns True on successful write, False on any error (missing
    bringup_status.json, malformed JSON, write failure). Best-effort:
    the caller treats False as "couldn't persist, skip graduation."
    """
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return False
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    components = data.get("components", [])
    if not isinstance(components, list):
        return False

    new_entry = spec.to_manifest_entry()
    updated = False
    for idx, existing in enumerate(components):
        if isinstance(existing, dict) and existing.get("name") == spec.name:
            # Preserve existing extras the decomposer set, but flip the
            # status to LATE_GRADUATE and refresh the spec fields.
            merged = dict(existing)
            merged.update(new_entry)
            components[idx] = merged
            updated = True
            break
    if not updated:
        components.append(new_entry)

    data["components"] = components
    try:
        status_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def list_late_graduate_components(demo_dir: Path) -> List[Dict[str, Any]]:
    """Return all components in the manifest with
    ``status == LATE_GRADUATE``. Useful for reruns and the family
    registry — distinguishes synthesis-discovered components from the
    decomposer's original static-analysis output."""
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return []
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    components = data.get("components", [])
    if not isinstance(components, list):
        return []
    return [c for c in components if isinstance(c, dict) and c.get("status") == LATE_GRADUATE_STATUS]


def mark_late_graduate_as_cpu_fallback(demo_dir: Path, component_name: str) -> bool:
    """Flip a late-graduate component to ``CPU_FALLBACK`` status when
    the per-component iterate exhausted its budget without converging.

    Mirrors the existing ``KERNEL_MISSING`` treatment but for the
    late-discovered case: the e2e synthesis can continue with the
    component on CPU, the kernel-gap report tracks it for future TTNN
    work.

    Returns True on successful write, False on any error.
    """
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return False
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    components = data.get("components", []) if isinstance(data, dict) else []
    if not isinstance(components, list):
        return False
    mutated = False
    for c in components:
        if isinstance(c, dict) and c.get("name") == component_name:
            c["status"] = "LATE_GRADUATE_CPU_FALLBACK"
            c["cpu_fallback_reason"] = (
                "per-component iterate budget exhausted; routed to CPU so "
                "e2e synthesis can continue without this component"
            )
            mutated = True
            break
    if not mutated:
        return False
    try:
        status_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


# ─── Late-graduate orchestrator ─────────────────────────────────────


def run_late_graduate(
    *,
    demo_dir: Path,
    spec: LateGraduateComponentSpec,
    component_iterate: Optional[Any] = None,
    max_iters: int = 5,
    pcc_target: float = 0.99,
) -> LateGraduateResult:
    """Graduate one late-discovered component.

    Steps:
      1. Add the spec to bringup_status.json with status=LATE_GRADUATE.
      2. Run ``component_iterate(demo_dir, name, max_iters, pcc_target)``
         to LLM-synth its stub + iterate to PCC ≥ pcc_target.
      3. On convergence: status stays LATE_GRADUATE; the e2e synthesis
         can use it.
         On failure: mark CPU fallback so e2e synthesis isn't blocked.

    The ``component_iterate`` callable is injectable so unit tests can
    drive the orchestrator without invoking the real LLM/pytest. Real
    callers pass a wrapper around :func:`llm_synth.synthesize_component`
    + the focused-pytest helper.

    Returns :class:`LateGraduateResult`. Never raises — every failure
    mode (manifest write fail, iterate raise) yields a result with
    ``converged=False`` and a diagnostic.
    """
    result = LateGraduateResult(component=spec.name, converged=False)

    if not add_late_graduate_to_manifest(demo_dir, spec):
        result.diagnostic = "failed to add to bringup_status.json"
        return result

    if component_iterate is None:
        result.diagnostic = "no component_iterate callable provided"
        return result

    try:
        sub_result = component_iterate(
            demo_dir=demo_dir,
            component_name=spec.name,
            max_iters=max_iters,
            pcc_target=pcc_target,
        )
    except Exception as exc:
        result.diagnostic = f"component_iterate raised {type(exc).__name__}: {exc}"
        mark_late_graduate_as_cpu_fallback(demo_dir, spec.name)
        result.fallback_to_cpu = True
        return result

    # Expected sub_result shape (loose): a dict-like with .converged,
    # .pcc, .iters_used. Be defensive — the injectable might return
    # a custom dataclass or dict; we accept anything readable.
    converged = bool(getattr(sub_result, "converged", False)) or bool(
        sub_result.get("converged", False) if isinstance(sub_result, dict) else False
    )
    pcc = getattr(sub_result, "pcc", None)
    if pcc is None and isinstance(sub_result, dict):
        pcc = sub_result.get("pcc")
    iters_used = getattr(sub_result, "iters_used", 0)
    if iters_used == 0 and isinstance(sub_result, dict):
        iters_used = sub_result.get("iters_used", 0)

    result.converged = converged
    result.pcc = pcc
    result.iters_used = int(iters_used or 0)

    if not converged:
        mark_late_graduate_as_cpu_fallback(demo_dir, spec.name)
        result.fallback_to_cpu = True
        result.diagnostic = (
            f"per-component iterate exhausted {result.iters_used}/{max_iters} "
            f"iters without reaching PCC ≥ {pcc_target}; routed to CPU fallback"
        )

    return result


# ─── Status filter helper for the iterate loop ──────────────────────


def _gradable_statuses() -> tuple:
    """Statuses an iterate-loop filter should recognize as "this needs
    a per-component PCC pass."

    Returns the tuple used by callers like
    ``c.get('status') in _gradable_statuses()``. Centralizes the list
    so adding a new status (e.g. LATE_GRADUATE) doesn't require
    finding and editing every NEW/ADAPT filter throughout the codebase.
    """
    return ("NEW", "ADAPT", LATE_GRADUATE_STATUS)


__all__ = [
    "LATE_GRADUATE_STATUS",
    "LateGraduateComponentSpec",
    "LateGraduateResult",
    "_gradable_statuses",
    "add_late_graduate_to_manifest",
    "list_late_graduate_components",
    "mark_late_graduate_as_cpu_fallback",
    "run_late_graduate",
]
