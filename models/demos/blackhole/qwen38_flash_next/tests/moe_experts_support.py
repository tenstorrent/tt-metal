# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the routed-expert program's device tests (x37): one layer's real cached BF4 experts on one die
of a 1x1 mesh, and the trace slope timer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import ttnn

CACHE_ROOT = Path(os.environ.get("TT_CACHE_PATH", "/home/ttuser/models/qwen38_cache")) / "caches" / "bf4-experts"


def cache_dir() -> Path:
    """The BF4 expert cache of 64 experts per die and an 8-bank ring that was read most recently (the TP=8 serving cache)."""
    best = None
    for manifest in CACHE_ROOT.glob("*/manifest.json"):
        identity = json.loads(manifest.read_text())["identity"]
        if identity["experts_per_device"] == 64 and identity["ring_size"] == 8:
            layer = manifest.parent / "backbone" / "layer-00" / "w2_dtype_BFLOAT4_B_layout_TILE.tensorbin"
            stamp = layer.stat().st_atime
            if best is None or stamp > best[0]:
                best = (stamp, manifest.parent)
    if best is None:
        raise RuntimeError(f"no 64-expert 8-ring BF4 cache under {CACHE_ROOT}")
    return best[1]


def load_layer_shard(mesh, layer: int, shard: int = 0):
    """``(w0_w1, w2)`` of backbone layer ``layer``: die ``shard``'s 64 experts of the 8-die cache on the one-die ``mesh``
    (``[8, 1, 64, 10, 320, 128]`` / ``[8, 1, 64, 5, 320, 128]`` BFLOAT4_B, DRAM height sharded over the 8 banks)."""
    root = cache_dir() / "backbone" / f"layer-{layer:02d}"
    out = []
    for name in ("w0_w1", "w2"):
        host = ttnn.load_tensor(root / f"{name}_dtype_BFLOAT4_B_layout_TILE.tensorbin")
        part = ttnn.get_device_tensors(host)[shard]
        out.append(ttnn.to_device(part, mesh, memory_config=part.memory_config()))
    return tuple(out)


def traced_wall_us(mesh, call, *, short: int = 4, long: int = 12, replays: int = 100, repeats: int = 3):
    """Wall us per call: the slope between a ``short``-call and a ``long``-call trace (the launch cost of the trace
    itself cancels), ``replays`` replays each, the best of ``repeats``; also (long trace)/long.  ``call`` runs once
    eagerly first (a program must be in the cache before a capture).  Returns ``(slope_us, per_call_us_of_long)``."""

    call()
    ttnn.synchronize_device(mesh)

    def capture(n):
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(n):
            call()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        return trace

    def wall(trace):
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = 1e30
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, time.perf_counter() - start)
        return best / replays * 1e6

    t_short, t_long = capture(short), capture(long)
    try:
        ws, wl = wall(t_short), wall(t_long)
    finally:
        ttnn.release_trace(mesh, t_short)
        ttnn.release_trace(mesh, t_long)
    return (wl - ws) / (long - short), wl / long
