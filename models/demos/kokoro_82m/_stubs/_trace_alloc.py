# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace-safe prealloc cache for the decode/vocode stages.

Trace capture forbids host->device writes, but the conv / STFT / source stubs create padding
`ttnn.zeros`, upsample `ttnn.ones`, and `ops._const` mask/interp tensors INSIDE their forward. At a
fixed frame capacity `Cf` these allocations are fixed-shape and value-stable, so we intercept them:
the FIRST (warmup) forward creates + caches each buffer (outside the trace), and every subsequent
call — including the one inside `begin/end_trace_capture` — returns the cached resident buffer, so the
captured program contains zero host writes.

All intercepted tensors are read-only in the graph (padding concatenated in, constants multiplied by,
masks compared against), so aliasing identical buffers across call sites is safe. Keying:
  * zeros / ones : (shape, dtype, layout, kind)                 -- identical fills alias freely
  * _const       : (shape, dtype, sha1(bytes))                  -- distinct constants stay distinct

Usage (see tt/pipeline.py decode/vocode stages):
    _trace_alloc.install()          # once: monkeypatch ttnn.zeros/ones + ops._const
    _trace_alloc.activate()         # cache-through ON (keep the warmed buffers)
    ... run forward once (warmup, populates cache) ...
    ... begin_trace_capture; run forward (cache hits, no writes); end_trace_capture ...
    _trace_alloc.deactivate()       # back to normal allocation
"""
from __future__ import annotations

import hashlib

import ttnn
from models.demos.kokoro_82m.tt import ops as _ops

_ACTIVE = False
_INSTALLED = False
_CACHE: dict = {}

_orig_zeros = None
_orig_ones = None
_orig_const = None


def _fill_key(shape, dtype, layout, kind):
    return (kind, tuple(int(s) for s in shape), str(dtype), str(layout))


def _zeros(*args, **kwargs):
    if not _ACTIVE:
        return _orig_zeros(*args, **kwargs)
    shape = args[0] if args else kwargs.get("shape")
    dtype = kwargs.get("dtype")
    layout = kwargs.get("layout")
    key = _fill_key(shape, dtype, layout, "zeros")
    t = _CACHE.get(key)
    if t is None:
        t = _orig_zeros(*args, **kwargs)
        _CACHE[key] = t
    return t


def _ones(*args, **kwargs):
    if not _ACTIVE:
        return _orig_ones(*args, **kwargs)
    shape = args[0] if args else kwargs.get("shape")
    dtype = kwargs.get("dtype")
    layout = kwargs.get("layout")
    key = _fill_key(shape, dtype, layout, "ones")
    t = _CACHE.get(key)
    if t is None:
        t = _orig_ones(*args, **kwargs)
        _CACHE[key] = t
    return t


def _const(device, t):
    if not _ACTIVE:
        return _orig_const(device, t)
    tc = t.detach().contiguous().float()
    h = hashlib.sha1(tc.cpu().numpy().tobytes()).hexdigest()
    key = ("const", tuple(int(s) for s in tc.shape), h)
    dev_t = _CACHE.get(key)
    if dev_t is None:
        dev_t = _orig_const(device, t)
        _CACHE[key] = dev_t
    return dev_t


def install():
    """Idempotently monkeypatch the three host-allocation entry points."""
    global _INSTALLED, _orig_zeros, _orig_ones, _orig_const
    if _INSTALLED:
        return
    _orig_zeros = ttnn.zeros
    _orig_ones = ttnn.ones
    _orig_const = _ops._const
    ttnn.zeros = _zeros
    ttnn.ones = _ones
    _ops._const = _const
    _INSTALLED = True


def activate():
    global _ACTIVE
    install()
    _ACTIVE = True


def deactivate():
    global _ACTIVE
    _ACTIVE = False


def reset():
    """Drop cached buffers (call when the fixed capacity Cf changes)."""
    _CACHE.clear()


def is_active() -> bool:
    return _ACTIVE
