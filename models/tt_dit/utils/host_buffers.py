# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Recycled host buffers for the YUV readback and clip assembly.

A fresh ``np.empty`` of 40-400 MB is new anonymous memory: every 4 KiB page faults on its first
write. On a serving box whose RAM is mostly page cache those faults can drop into direct reclaim,
and the planar assembly of one 43 MB wave went from 5 ms to 277 ms p50 and 1 s max on blx04
(11-15, ``planar_concat_cpp`` into a fresh buffer vs into one already mapped). One readback wave
arrives every ~120 ms, so the slow ones queue up on the single finish worker and stall the clip.

:func:`empty_uint8` hands back a buffer a previous clip released instead. A slab is free once
nothing but this pool refers to it: numpy views and ``torch.from_numpy`` all keep their base
alive, so CPython's refcount tells exactly when the last user let go. Contents are undefined,
as with ``np.empty``. ``TT_DIT_HOST_BUFFER_POOL_MB`` caps the bytes the pool holds, live and idle
(default 3072; 0 turns the pool off and every call is a plain ``np.empty``; so is any buffer that
does not fit under the cap once the idle slabs are dropped).

Reuse alone still leaves the first clips after a model load or a length switch on fresh memory,
and every clip whose earlier buffers readback or export still hold: on blx04 (11-20, 8 GB free,
476 GB page cache) those assemblies took 0.2-1.9 s against 0.05 s. So each hand-out also keeps
``TT_DIT_HOST_BUFFER_SPARES`` (default 1) idle slabs of its size mapped ahead: a background thread
allocates and touches them, within the cap, while the next clip denoises. :func:`reserve` does
the same for a size a caller knows is coming (the first clip at a new length).
"""

from __future__ import annotations

import math
import os
import sys
import threading

import numpy as np

_MIN_BYTES = 1 << 20  # small buffers fault in quickly; not worth holding
_PAGE = 4096
_FILL_WAIT_S = 5.0  # past this, a waiter stops waiting on a fill and maps its own slab
_lock = threading.Lock()
_slabs: list[np.ndarray] = []  # flat uint8 owners, oldest first; live or idle
_filling: dict[int, threading.Event] = {}  # slab size -> set once its background fill ends
_filling_bytes = 0  # slabs a fill has allocated but not yet put in the pool


def _spares() -> int:
    try:
        return max(0, int(os.environ.get("TT_DIT_HOST_BUFFER_SPARES", "1")))
    except ValueError:
        return 1


def _max_idle_bytes() -> int:
    try:
        return int(os.environ.get("TT_DIT_HOST_BUFFER_POOL_MB", "3072")) << 20
    except ValueError:
        return 3072 << 20


def _refs(slab: np.ndarray) -> int:
    return sys.getrefcount(slab)


def _idle_refs() -> int:
    """What ``_refs`` reads for a slab only the pool holds: the list entry, the loop name and the call."""
    probe = [np.empty(1, dtype=np.uint8)]
    for slab in probe:
        return _refs(slab)
    raise AssertionError


_IDLE_REFS = _idle_refs()


def _make_room(n: int, cap: int) -> bool:
    """Under ``_lock``: drop the oldest idle slabs of other sizes until ``n`` more bytes fit under the cap.

    Live slabs count too: they come back to the pool when released. Returns whether ``n`` fits.
    """
    idle = []
    held = _filling_bytes
    for slab in _slabs:
        held += slab.size
        if slab.size != n and _refs(slab) == _IDLE_REFS:
            idle.append(slab)
    spare = held + n - cap
    if spare > sum(s.size for s in idle):  # dropping them would not make room
        return False
    drop = set()
    for slab in idle:
        if spare <= 0:
            break
        spare -= slab.size
        drop.add(id(slab))
    if drop:
        # By identity: ``list.remove`` would compare arrays element by element.
        _slabs[:] = [s for s in _slabs if id(s) not in drop]
    return True


def _idle_count(n: int) -> int:
    count = 0
    for slab in _slabs:
        if slab.size == n and _refs(slab) == _IDLE_REFS:
            count += 1
    return count


def _fill(n: int, count: int) -> None:
    """Allocate and touch slabs of ``n`` bytes until ``count`` are idle, or the cap is reached."""
    global _filling_bytes
    try:
        while True:
            with _lock:
                if _idle_count(n) >= count or not _make_room(n, _max_idle_bytes()):
                    return
                _filling_bytes += n
            slab = None
            try:
                slab = np.empty(n, dtype=np.uint8)
                slab[::_PAGE] = 0  # one write per page maps it; numpy drops the GIL for it
            finally:
                with _lock:
                    _filling_bytes -= n
                    if slab is not None:
                        _slabs.append(slab)
                    # Wake whoever waits for this size now, not after the whole fill.
                    _filling[n].set()
                    _filling[n] = threading.Event()
            del slab
    finally:
        with _lock:
            _filling.pop(n).set()


def _top_up(n: int, count: int) -> None:
    """Start a background fill of ``n``-byte slabs unless one runs or ``count`` are already idle."""
    if count <= 0:
        return
    with _lock:
        if n in _filling or _idle_count(n) >= count:
            return
        _filling[n] = threading.Event()
    threading.Thread(target=_fill, args=(n, count), name="host-buffer-fill", daemon=True).start()


def reserve(shape, count: int | None = None) -> None:
    """Map ``count`` idle slabs for ``empty_uint8(shape)`` in the background (default: the spares setting)."""
    n = math.prod(int(d) for d in shape)
    if _MIN_BYTES <= n <= _max_idle_bytes():
        _top_up(n, _spares() if count is None else count)


def empty_uint8(shape) -> np.ndarray:
    """``np.empty(shape, np.uint8)``, from a released slab of the same size when there is one."""
    n = math.prod(int(d) for d in shape)
    cap = _max_idle_bytes()
    if n < _MIN_BYTES or n > cap:
        return np.empty(shape, dtype=np.uint8)
    stuck = None
    while True:
        with _lock:
            for slab in _slabs:
                if slab.size == n and _refs(slab) == _IDLE_REFS:
                    break
            else:
                slab = None
                filling = _filling.get(n)
                if filling is None or filling is stuck:
                    pooled = _make_room(n, cap)
                    slab = np.empty(n, dtype=np.uint8)
                    if pooled:
                        _slabs.append(slab)
        if slab is not None:
            break
        # A fill of this size is mapping a slab now: waiting for it beats faulting in another.
        if not filling.wait(_FILL_WAIT_S):
            stuck = filling
    out = slab.reshape(shape)
    del slab
    _top_up(n, _spares())
    return out
