# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for older transformers releases."""

from __future__ import annotations

from loguru import logger


class _DynamicCacheLayerCompat:
    """Expose key/value tensors via the transformers >=4.57 CacheLayer API."""

    __slots__ = ("_cache", "_layer_idx")

    def __init__(self, cache, layer_idx: int) -> None:
        self._cache = cache
        self._layer_idx = layer_idx

    @property
    def keys(self):
        return self._cache.key_cache[self._layer_idx]

    @property
    def values(self):
        return self._cache.value_cache[self._layer_idx]


class _DynamicCacheLayersCompat:
    __slots__ = ("_cache",)

    def __init__(self, cache) -> None:
        self._cache = cache

    def __getitem__(self, layer_idx: int) -> _DynamicCacheLayerCompat:
        return _DynamicCacheLayerCompat(self._cache, layer_idx)


def apply_transformers_cache_compat() -> None:
    """Backport ``DynamicCache.layers`` for transformers releases before 4.57."""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    if getattr(DynamicCache, "__acestep_layers_compat_patched__", False):
        return

    if "layers" in DynamicCache.__dict__:
        # Native transformers >=4.57 API is already present.
        return

    def _layers(self):
        return _DynamicCacheLayersCompat(self)

    DynamicCache.layers = property(_layers)
    DynamicCache.__acestep_layers_compat_patched__ = True
    logger.debug("[compat] Patched transformers.cache_utils.DynamicCache.layers for older transformers")
