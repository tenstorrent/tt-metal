"""HOT/COLD profiler.

For a given model + workload, identify which NEW components are
actually INVOKED during the demo's forward pass (HOT) vs. components
that exist in the model but are never called by this particular
workload (COLD).

The distinction matters because Phase 2 / standalone PCC requirements
should only apply to HOT components. COLD components that live on
CPU fallback are not a regression -- they would also be idle on
CPU in the torch reference. Spending LLM budget porting them to TT
is wasted effort.

Generic and model-agnostic. The profiler does not reference any
specific HF class names -- it uses the same _resolve_submodule
chain that ``capture_inputs.py`` uses, then attaches forward hooks
and invokes the model via ``capture_drivers.try_capture_drivers``
to fire the demo's actual code path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_HOT = "HOT"
_COLD = "COLD"
_UNRESOLVED = "UNRESOLVED"


def _attach_hook(submodule: Any, name: str, fired: set) -> Optional[Any]:
    """Register a forward pre-hook that records the component name when
    the submodule's forward is invoked. Returns the hook handle (caller
    is responsible for removing it) or None if registration fails."""
    try:
        return submodule.register_forward_pre_hook(lambda _mod, _inp, _name=name: fired.add(_name))
    except Exception:
        return None


def profile_hot_cold(
    *,
    model: Any,
    components: List[Dict[str, Any]],
    demo_dir: Path,
    sample_input: Any,
) -> Dict[str, str]:
    """Profile which components fire during a sample forward pass.

    Args:
        model: A loaded HF model (already on CPU / eval mode).
        components: List of {"name": ..., ...} dicts from bringup_status.json.
            Only entries with status == "NEW" are profiled; others
            are returned as-is (REUSE / ADAPT components are out of scope
            for the auto-iterate loop's graduation work).
        demo_dir: The model's demo directory (for _resolve_submodule
            candidate-path lookup).
        sample_input: A torch tensor or appropriate object to drive
            model.forward(). Typically ``torch.randn(1, 3, H, W)`` for
            image-mode workloads.

    Returns:
        ``{comp_name: "HOT" | "COLD" | "UNRESOLVED"}`` for each NEW
        component. "UNRESOLVED" means we couldn't find the submodule
        in the model at all -- treated conservatively as if HOT so the
        tool doesn't silently dead-code something it should port.
    """
    from .capture_inputs import _resolve_submodule

    fired: set = set()
    hooks: List[Any] = []
    resolved: Dict[str, str] = {}
    unresolved: List[str] = []

    new_components = [c for c in components if c.get("status") == "NEW"]
    for comp in new_components:
        name = comp.get("name")
        if not name:
            continue
        result = _resolve_submodule(model, name, demo_dir=demo_dir)
        if result is None:
            unresolved.append(name)
            continue
        submodule, _path = result
        handle = _attach_hook(submodule, name, fired)
        if handle is not None:
            hooks.append(handle)
            resolved[name] = "ok"

    try:
        _invoke_model_for_profile(model, sample_input)
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    out: Dict[str, str] = {}
    for comp in new_components:
        name = comp.get("name")
        if not name:
            continue
        if name in unresolved:
            out[name] = _UNRESOLVED
        elif name in fired:
            out[name] = _HOT
        else:
            out[name] = _COLD
    return out


def _invoke_model_for_profile(model: Any, sample_input: Any) -> None:
    """Best-effort invocation of model.forward to fire hooks. Tries
    multiple invocation patterns since HF model signatures vary. Never
    raises -- if every path fails, the caller still gets a fully-COLD
    classification, which is the conservative outcome (the caller will
    notice "0 HOT components" and re-investigate).

    Chain (in order):
      Layer 1: model(pixel_values=sample) — vanilla vision encoders
      Layer 2: try_capture_drivers       — generic Session/init patterns
      Layer 3: model(sample)             — vanilla bare-tensor input
      Layer 4: learned invoker registry  — LLM-drafted invokers from
               ``auto_hot_cold_invoker_onboard``. Loaded from
               ``learned_invokers/*.py`` at the start of every profile
               call so persisted invokers are picked up automatically.
    """
    try:
        import torch

        # Load learned drivers + invokers so any registered custom paths
        # are available before we try the chain.
        try:
            from .auto_capture_driver_onboard import load_learned_drivers

            load_learned_drivers()
        except Exception:
            pass
        try:
            from .auto_hot_cold_invoker_onboard import load_learned_invokers, resolve_custom_invoker

            load_learned_invokers()
        except Exception:
            resolve_custom_invoker = None  # type: ignore

        with torch.no_grad():
            try:
                model(pixel_values=sample_input)
                return
            except Exception:
                pass
            try:
                from .capture_drivers import try_capture_drivers

                ok, _attempts = try_capture_drivers(model, sample_input)
                if ok:
                    return
            except Exception:
                pass
            try:
                model(sample_input)
                return
            except Exception:
                pass
            # Layer 4: LLM-drafted invoker (registered for this model class)
            if resolve_custom_invoker is not None:
                try:
                    custom = resolve_custom_invoker(model)
                    if custom is not None:
                        custom(model, sample_input)
                        return
                except Exception:
                    pass
    except ImportError:
        return


def make_sample_input(*, batch: int = 1, channels: int = 3, height: int = 1024, width: int = 1024) -> Any:
    """Construct a default sample input tensor for image-style workloads.

    Generic shape suitable for most vision encoders. Callers with
    non-image workloads (audio, text) should construct their own input
    and pass it to ``profile_hot_cold`` directly. Centralizing the
    default here so the CLI command and tests share the same expression.
    """
    import torch

    return torch.randn(batch, channels, height, width)


def summarize_hot_cold(classification: Dict[str, str]) -> Dict[str, List[str]]:
    """Group classifications by category for clean reporting.

    Returns ``{"HOT": [...], "COLD": [...], "UNRESOLVED": [...]}``
    with components sorted within each bucket."""
    buckets: Dict[str, List[str]] = {_HOT: [], _COLD: [], _UNRESOLVED: []}
    for comp, kind in classification.items():
        buckets.setdefault(kind, []).append(comp)
    for k in buckets:
        buckets[k].sort()
    return buckets
