"""Compatibility for ACE-Step HF modeling vs tt-metal's pinned transformers (4.53.x).

Upstream ACE-Step DiT code reads cross-attention KV as::

    curr_past_key_value.layers[layer_idx].keys

That API exists from transformers ~4.55 (``CacheLayer`` / ``DynamicCache.layers``). tt-metal pins
``transformers==4.53.0``, where ``DynamicCache`` only exposes ``key_cache`` / ``value_cache``.
"""

from __future__ import annotations

from typing import Any


class _CacheLayerKeyValueView:
    __slots__ = ("keys", "values")

    def __init__(self, keys: Any, values: Any) -> None:
        self.keys = keys
        self.values = values


def _dynamic_cache_layers_property(self: Any) -> list[_CacheLayerKeyValueView]:
    key_cache = getattr(self, "key_cache", None)
    value_cache = getattr(self, "value_cache", None)
    if key_cache is None or value_cache is None:
        raise AttributeError("DynamicCache has no key_cache/value_cache; cannot build ACE-Step layers compat view.")
    n = max(len(key_cache), len(value_cache))
    return [_CacheLayerKeyValueView(key_cache[i], value_cache[i]) for i in range(n)]


def _transformers_version_tuple() -> tuple[int, ...]:
    import transformers

    parts: list[int] = []
    for piece in transformers.__version__.split("."):
        if not piece or not piece[0].isdigit():
            break
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def apply_transformers_cache_compat() -> None:
    """Patch ``DynamicCache`` when the legacy list-based KV API is in use."""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    if getattr(DynamicCache, "_acestep_layers_compat_applied", False):
        return

    # ACE-Step modeling targets transformers >= 4.55 (CacheLayer / DynamicCache.layers).
    if _transformers_version_tuple() >= (4, 55):
        DynamicCache._acestep_layers_compat_applied = True
        return

    DynamicCache.layers = property(_dynamic_cache_layers_property)  # type: ignore[attr-defined]
    DynamicCache._acestep_layers_compat_applied = True
