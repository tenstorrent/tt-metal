# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared opt-in "trace-prep" state for Kokoro metal-trace capture.

Metal-trace capture (``ttnn.begin_trace_capture``) forbids host->device writes. Several Kokoro
modules upload constant tensors from host on every forward — conv prepared weights (tt_conv), the
BERT embedding id tensors (tt_custom_albert), the BiLSTM reversal matrix / zero states (tt_lstm).
Those uploads are the same tensors every call, so when trace-prep is enabled each site uploads once,
caches the device tensor here, and reuses it — making the graph capturable. Default OFF: every
existing (non-traced) caller is byte-for-byte unchanged.

The cache is keyed by whatever discriminator the call site builds (usually ``id(params)`` plus a
shape/dtype/content signature). Cached tensors live on the device that created them; call
:func:`clear_trace_weight_prep_cache` before closing that device.
"""

from __future__ import annotations

import ttnn

_ENABLED = False
_CACHE: dict = {}


def set_trace_weight_prep(enabled: bool) -> None:
    """Enable/disable trace-prep caching across all Kokoro modules. Off by default."""
    global _ENABLED
    _ENABLED = bool(enabled)


def trace_weight_prep_enabled() -> bool:
    """Whether trace-prep caching is currently enabled."""
    return _ENABLED


def clear_trace_weight_prep_cache() -> None:
    """Drop all cached device tensors. Call before closing the owning device."""
    _CACHE.clear()


def prep_cache_get(key):
    """Return the cached value for ``key`` (or ``None``). Only meaningful when enabled."""
    return _CACHE.get(key)


def prep_cache_set(key, value) -> None:
    """Store ``value`` under ``key`` in the trace-prep cache."""
    _CACHE[key] = value


def traced_zeros(shape, *, dtype, device, memory_config, key):
    """A zeros tensor that is trace-capturable.

    ``ttnn.zeros(device=...)`` uploads from host — illegal inside trace capture. Callers whose zeros
    are *consumed* (freed/reassigned) can't just cache-and-reuse, so under trace weight prep we upload
    one cached zero *template* per (shape, dtype) and return a fresh ``ttnn.clone`` of it each call
    (clone is a device-only op → trace-clean, and the caller may freely free the clone). Prep off = a
    plain ``ttnn.zeros`` (original behaviour). Byte-identical either way (all zeros).
    """
    if _ENABLED:
        tmpl = _CACHE.get(key)
        if tmpl is None:
            tmpl = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
            _CACHE[key] = tmpl
        return ttnn.clone(tmpl, memory_config=memory_config)
    return ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
