"""Pure LoRA introspection helpers reusable across runtimes and UIs."""

import math
from collections.abc import Callable
from typing import Any


def collect_adapter_names(decoder: Any) -> list[str]:
    """Best-effort adapter name discovery across PEFT runtime variants."""

    def _extract_names(value: Any) -> list[str]:
        names: list[str] = []

        def _append_name(name: Any) -> None:
            if isinstance(name, str) and name and name not in names:
                names.append(name)

        def _walk(obj: Any) -> None:
            if obj is None:
                return
            if isinstance(obj, str):
                _append_name(obj)
                return
            if isinstance(obj, dict):
                for key in obj.keys():
                    _append_name(key)
                return
            if isinstance(obj, (list, tuple, set)):
                for item in obj:
                    _walk(item)
                return
            if hasattr(obj, "keys") and callable(obj.keys):
                try:
                    for key in obj.keys():
                        _append_name(key)
                except Exception:
                    pass
            if hasattr(obj, "adapters"):
                _walk(getattr(obj, "adapters"))
            if hasattr(obj, "adapter_names"):
                _walk(getattr(obj, "adapter_names"))
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                try:
                    _walk(obj.to_dict())
                except Exception:
                    pass

        _walk(value)
        return list(dict.fromkeys(names))

    ordered: list[str] = []
    source_groups: list[list[str]] = []

    if hasattr(decoder, "get_adapter_names") and callable(decoder.get_adapter_names):
        try:
            names_value = decoder.get_adapter_names()
            source_groups.append(_extract_names(names_value() if callable(names_value) else names_value))
        except Exception:
            pass

    for attr in ("active_adapter", "active_adapters", "peft_config"):
        if not hasattr(decoder, attr):
            continue
        try:
            value = getattr(decoder, attr)
            source_groups.append(_extract_names(value() if callable(value) else value))
        except Exception:
            pass

    for group in source_groups:
        for name in group:
            if name not in ordered:
                ordered.append(name)
    return ordered


def is_lora_like_module(name: str, module: Any) -> bool:
    """Conservative LoRA module detection for mixed PEFT implementations."""
    name_l = name.lower()
    cls_l = module.__class__.__name__.lower()
    mod_l = module.__class__.__module__.lower()
    has_lora_signals = (
        "lora" in name_l
        or "lora" in cls_l
        or ("peft" in mod_l and "lora" in mod_l)
        or hasattr(module, "lora_A")
        or hasattr(module, "lora_B")
    )
    has_scaling_api = hasattr(module, "scaling") or hasattr(module, "set_scale") or hasattr(module, "scale_layer")
    return has_lora_signals and has_scaling_api


def read_adapter_value(value: Any, adapter: str) -> Any:
    """Read adapter-specific value from mapping-like or scalar containers."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(adapter)
    if hasattr(value, "keys") and callable(value.keys):
        try:
            return value.get(adapter)
        except Exception:
            return None
    if isinstance(value, (int, float)):
        return value
    return None


def is_peft_factor_set_scale_module(module: Any) -> bool:
    """Detect modules where set_scale(adapter, factor) semantics are expected."""
    return hasattr(module, "set_scale") and hasattr(module, "lora_alpha") and hasattr(module, "r")


def get_peft_initial_scale(
    module: Any,
    adapter: str,
    debug_hook: Callable[[str], None] | None = None,
) -> float | None:
    """Return PEFT LoRA baseline scale (alpha/r or alpha/sqrt(r)) for adapter."""
    try:
        alpha = read_adapter_value(getattr(module, "lora_alpha", None), adapter)
        r_val = read_adapter_value(getattr(module, "r", None), adapter)
        if not isinstance(alpha, (int, float)) or not isinstance(r_val, (int, float)) or not r_val:
            return None
        use_rslora_raw = getattr(module, "use_rslora", False)
        use_rslora = (
            bool(use_rslora_raw.get(adapter, False)) if isinstance(use_rslora_raw, dict) else bool(use_rslora_raw)
        )
        return (alpha / math.sqrt(r_val)) if use_rslora else (alpha / r_val)
    except Exception as exc:
        if debug_hook is not None:
            debug_hook(f"Failed to compute initial scale (adapter={adapter}, err={exc})")
        return None
