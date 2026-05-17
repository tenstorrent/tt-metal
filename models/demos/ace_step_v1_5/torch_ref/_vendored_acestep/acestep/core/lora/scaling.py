"""Pure LoRA scale application using registry entries."""

import traceback
from collections.abc import Callable
from typing import Any

MIN_PREV_SCALE = 1e-12


def _inc(store: dict[str, int], key: str) -> None:
    store[key] = store.get(key, 0) + 1


def apply_scale_to_adapter(
    registry: dict[str, dict[str, Any]],
    scale_state: dict[tuple[int, str, str], float],
    adapter_name: str,
    scale: float,
    warn_hook: Callable[[str], None] | None = None,
    debug_hook: Callable[[str], None] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Apply scale to one adapter and return `(modified_count, report)`."""
    meta = registry.get(adapter_name)
    if not meta:
        report = {
            "adapter": adapter_name,
            "modified_total": 0,
            "modified_by_kind": {},
            "skipped_by_kind": {"no_registry": 1},
        }
        return 0, report

    modified = 0
    modified_by_kind: dict[str, int] = {}
    skipped_by_kind: dict[str, int] = {}

    for target in meta.get("targets", []):
        module = target.get("module")
        kind = target.get("kind")
        kind_key = kind if isinstance(kind, str) and kind else "unknown_kind"
        module_name = target.get("module_name")
        if module is None:
            _inc(skipped_by_kind, kind_key)
            continue

        try:
            if kind == "scaling_dict":
                adapter = target.get("adapter")
                if adapter not in module.scaling:
                    _inc(skipped_by_kind, kind_key)
                    continue
                module.scaling[adapter] = target.get("base_scale", module.scaling[adapter]) * scale
                modified += 1
                _inc(modified_by_kind, kind_key)
            elif kind == "set_scale_factor":
                base_factor = target.get("base_factor", None)
                if isinstance(base_factor, (int, float)):
                    module.set_scale(adapter_name, base_factor * scale)
                    modified += 1
                    _inc(modified_by_kind, kind_key)
                else:
                    _inc(skipped_by_kind, "set_scale_factor_unanchored")
                    if warn_hook:
                        warn_hook(
                            f"Skipping set_scale_factor target without anchor "
                            f"(adapter={adapter_name}, module={target.get('module_name')})"
                        )
            elif kind == "set_scale_unknown":
                base_scale = target.get("base_scale", None)
                if isinstance(base_scale, (int, float)):
                    module.set_scale(adapter_name, base_scale * scale)
                    modified += 1
                    _inc(modified_by_kind, kind_key)
                else:
                    _inc(skipped_by_kind, kind_key)
                    if warn_hook:
                        warn_hook(
                            f"Skipping set_scale target with unknown semantics and no base "
                            f"(adapter={adapter_name}, module={target.get('module_name')})"
                        )
                    if debug_hook:
                        debug_hook(
                            f"Skipped unanchored set_scale target "
                            f"(adapter={adapter_name}, module={target.get('module_name')})"
                        )
            elif kind == "scale_layer":
                # scale_state tracks layer-scaling semantics (absolute desired layer scale).
                base_scale = target.get("base_scale", None)
                desired = (base_scale * scale) if isinstance(base_scale, (int, float)) else scale
                state_key = (id(module), kind_key, adapter_name)
                if hasattr(module, "unscale_layer"):
                    # Record the applied absolute desired scale so state stays coherent.
                    module.unscale_layer()
                    module.scale_layer(desired)
                    scale_state[state_key] = float(desired)
                    modified += 1
                    _inc(modified_by_kind, kind_key if base_scale is not None else "scale_layer_fallback")
                elif base_scale is None:
                    _inc(skipped_by_kind, "scale_layer_unanchored")
                    if warn_hook:
                        warn_hook(
                            f"Skipping unanchored scale_layer target without unscale_layer "
                            f"(adapter={adapter_name}, module={target.get('module_name')})"
                        )
                else:
                    prev = scale_state.get(state_key)
                    module.scale_layer(
                        desired / prev if isinstance(prev, (int, float)) and prev > MIN_PREV_SCALE else desired
                    )
                    scale_state[state_key] = float(desired)
                    modified += 1
                    _inc(modified_by_kind, kind_key)
            elif kind == "scaling_scalar":
                base_scale = target.get("base_scale", None)
                if not isinstance(base_scale, (int, float)):
                    current_scaling = getattr(module, "scaling", None)
                    if not isinstance(current_scaling, (int, float)):
                        _inc(skipped_by_kind, kind_key)
                        if warn_hook:
                            warn_hook(
                                f"Skipping scaling_scalar target with non-numeric scaling "
                                f"(adapter={adapter_name}, module={module_name})"
                            )
                        continue
                    base_scale = float(current_scaling)
                module.scaling = base_scale * scale
                modified += 1
                _inc(modified_by_kind, kind_key)
        except Exception as exc:
            _inc(skipped_by_kind, kind_key)
            if warn_hook:
                warn_hook(
                    f"Failed to apply LoRA scale target "
                    f"(adapter={adapter_name}, module={module_name}, kind={kind_key}, err={exc})"
                )
            if debug_hook:
                err_tb = traceback.format_exc()
                debug_hook(
                    f"Scale application exception for target "
                    f"(adapter={adapter_name}, module={module_name}, kind={kind_key}, err={exc}, tb={err_tb})"
                )

    report = {
        "adapter": adapter_name,
        "modified_total": modified,
        "modified_by_kind": modified_by_kind,
        "skipped_by_kind": skipped_by_kind,
    }
    return modified, report
