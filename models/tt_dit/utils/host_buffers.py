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
"""

from __future__ import annotations

import math
import os
import sys
import threading

import numpy as np

_MIN_BYTES = 1 << 20  # small buffers fault in quickly; not worth holding
_lock = threading.Lock()
_slabs: list[np.ndarray] = []  # flat uint8 owners, oldest first; live or idle


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


def empty_uint8(shape) -> np.ndarray:
    """``np.empty(shape, np.uint8)``, from a released slab of the same size when there is one."""
    n = math.prod(int(d) for d in shape)
    cap = _max_idle_bytes()
    if n < _MIN_BYTES or n > cap:
        return np.empty(shape, dtype=np.uint8)
    with _lock:
        idle = []
        held = 0
        for slab in _slabs:
            held += slab.size
            if _refs(slab) == _IDLE_REFS:
                if slab.size == n:
                    return slab.reshape(shape)
                idle.append(slab)
        # Drop the oldest idle slabs of other sizes until the new one fits under the cap. Live
        # slabs count too: they come back to the pool when released.
        spare = held + n - cap
        drop = set()
        if spare <= sum(s.size for s in idle):  # else dropping them would not make room
            for slab in idle:
                if spare <= 0:
                    break
                spare -= slab.size
                drop.add(id(slab))
        if drop:
            # By identity: ``list.remove`` would compare arrays element by element.
            _slabs[:] = [s for s in _slabs if id(s) not in drop]
        del idle
        slab = np.empty(n, dtype=np.uint8)
        if spare <= 0:
            _slabs.append(slab)
        return slab.reshape(shape)
