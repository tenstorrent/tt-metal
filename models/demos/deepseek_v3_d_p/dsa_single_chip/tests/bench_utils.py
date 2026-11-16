# SPDX-License-Identifier: Apache-2.0
"""Shared harness: open one chip, build inputs, time ops with trace replay, compare outputs."""

from __future__ import annotations

import statistics
import time

import torch

import ttnn


def open_device():
    """One Blackhole chip as a 1x1 mesh (the ops under test run per chip; some need mesh semantics)."""
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=64 * 1024 * 1024)


def close_device(dev):
    ttnn.close_mesh_device(dev)


def time_trace(device, fn, iters: int = 50, repeats: int = 7) -> float:
    """Median microseconds per call of ``fn`` (called ``iters`` times inside one captured trace).

    ``fn`` must not allocate unbounded memory per call; outputs it returns are ignored."""
    fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(iters):
        fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)  # warm
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) / iters * 1e6)
    ttnn.release_trace(device, tid)
    return statistics.median(samples)


def to_dev(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=mem)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    if torch.equal(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def report(name: str, ref_us: float, new_us: float, pcc_val: float | None = None, extra: str = "") -> str:
    p = "" if pcc_val is None else f" pcc={pcc_val:.6f}"
    return f"[{name}] existing={ref_us:.2f}us new={new_us:.2f}us speedup={ref_us / new_us:.2f}x{p} {extra}"
