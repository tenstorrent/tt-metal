# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared 'L1 clean' check for decode-path variants (use from any harness):

    from models.demos.blackhole.deepseek_v41_flash.tests import l1_clean as L1C
    before = L1C.snapshot(md, "before")
    ... run one decode step / one replay of a captured trace ...
    L1C.assert_clean(md, before, "after")

snapshot(): the allocator view of L1 on device 0 of the mesh (ttnn.get_memory_view: allocated bytes / number of allocated buffers per bank, free, largest free block ...) and, when the
HYBRID allocator is available, the occupied L1 ranges of a bank core (ttnn.experimental_get_l1_occupied_ranges); every numeric field of the view is kept.
assert_clean(): (1) the allocated L1 bytes and the number of live L1 buffers after the step equal those before it (everything allocated inside the step was deallocated), (2) no occupied range
above PRELL_END (the end of the prefill MoE static CB region, 685968 B) except the known persistent buffers (PERSISTENT_OK: fuse-tail landing / receive buffers, listed by tt/fuse_tail_ar.py).
Prints one line "L1CLEAN <tag> ..." so the logs can be grepped; raises AssertionError on a violation unless L1C_SOFT=1.
"""

import os

import ttnn

PRELL_END = 685968  # end of the prefill MoE static CB region (dsv4-logs/fusedec/HANDOFF.md)
PERSISTENT_OK = "fuse-tail landing buffers (40 cores x 5 compact tiles) + 7-slot receive buffers (L1_SMALL) + their global semaphores"


def _view(md):
    dev = md.get_devices()[0] if hasattr(md, "get_devices") else md
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    return {k: getattr(mv, k) for k in dir(mv) if not k.startswith("_") and isinstance(getattr(mv, k), (int, float))}


def _ranges(md):
    try:
        coord = ttnn.MeshCoordinate(0, 0)
        r = ttnn.experimental_get_l1_occupied_ranges(md, coord, ttnn.CoreCoord(1, 2))
        return sorted((int(a), int(b)) for a, b in r)
    except Exception as e:  # allocator without ranges support
        return f"unavailable: {type(e).__name__}"


def snapshot(md, tag=""):
    s = dict(view=_view(md), ranges=_ranges(md), tag=tag)
    print(f"L1CLEAN snapshot {tag}: {s['view']}", flush=True)
    return s


def assert_clean(md, before, tag=""):
    now = snapshot(md, tag)
    bad = []
    for k, v in before["view"].items():
        if "alloc" in k and now["view"].get(k) != v:
            bad.append(f"{k}: {v} -> {now['view'].get(k)}")
    if isinstance(now["ranges"], list):
        high = [
            r
            for r in now["ranges"]
            if r[1] > PRELL_END and r not in (before["ranges"] if isinstance(before["ranges"], list) else [])
        ]
        if high:
            bad.append(f"new occupied L1 ranges above {PRELL_END}: {high[:6]}")
    print(
        f"L1CLEAN {tag}: {'CLEAN' if not bad else 'VIOLATION ' + '; '.join(bad)} (persistent allowed: {PERSISTENT_OK})",
        flush=True,
    )
    if bad and os.environ.get("L1C_SOFT") != "1":
        raise AssertionError("L1 not clean: " + "; ".join(bad))
    return not bad
