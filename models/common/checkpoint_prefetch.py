# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Background page-table population for memory-mapped checkpoints during cold weight-cache builds.

HF state dicts are lazily mapped safetensors. On a cold build every weight is read once while being converted, and
each 4 KB page is mapped on first touch: on a 27B checkpoint those demand faults made the host-side weight transposes
~3x slower than the copy itself (threads contending on the address-space lock). Populating the page tables ahead of
use with madvise(MADV_POPULATE_READ) from a few background threads, in layer order, takes that off the critical path.
It only maps pages (reading them from disk if they are not cached); it never changes any data.
"""

import ctypes
import re
from concurrent.futures import ThreadPoolExecutor

import torch
from loguru import logger

_MADV_POPULATE_READ = 22  # Linux >= 5.14
_PAGE = 4096
_libc = None


def _populate(ptr, nbytes):
    start = ptr & ~(_PAGE - 1)
    # Failures (old kernel, region already unmapped) only lose the optimization.
    _libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(nbytes + (ptr - start)), _MADV_POPULATE_READ)


def _layer_order(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return (int(m.group(1)) if m else -1, name)


def prefetch_state_dict(state_dict, threads=8, min_bytes=1 << 20):
    """Start populating the page tables of every large CPU tensor in `state_dict` (layer order); returns at once."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    regions, seen = [], set()
    for name in sorted(state_dict, key=_layer_order):
        t = state_dict[name]
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
            continue
        storage = t.untyped_storage()
        ptr, nbytes = storage.data_ptr(), storage.nbytes()
        if nbytes < min_bytes or ptr in seen:
            continue
        seen.add(ptr)
        regions.append((ptr, nbytes))
    if not regions:
        return None
    pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="ckpt-prefetch")
    for ptr, nbytes in regions:
        pool.submit(_populate, ptr, nbytes)
    pool.shutdown(wait=False)
    logger.info(f"Prefetching {sum(n for _, n in regions) / 1e9:.1f} GB of checkpoint pages in the background")
    return pool
