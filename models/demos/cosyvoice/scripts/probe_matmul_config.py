# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Is the batch-1 decode matmul core-limited, and does a core grid fix it?

The decode step moves 352 MB of weights per token and does it in 8.25 ms -- about
42 GB/s, roughly 8 % of what this part can deliver. Two measurements say the
shortfall is not bandwidth: `bfloat8_b` weights, which halve the bytes, measured
exactly 1.00x at both 27 and 42 GB/s. If halving the traffic changes nothing, the
matmul is not waiting on DRAM; the likely cause is that a `[1, K] x [K, N]` product
lands on few cores.

That is worth one contained experiment before either attempting a sharded rewrite of
every linear or declaring the RTF target out of reach, because the two conclusions
differ by a lot of work.

Each shape below is one of the four linears an AR decoder layer issues.

    python models/demos/cosyvoice/scripts/probe_matmul_config.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

# (in, out) for qkv, out-proj, ffn-up, ffn-down at d_model = 1024, d_ff = 4096
SHAPES = [(1024, 3072), (1024, 1024), (1024, 4096), (4096, 1024)]


def timed(device, fn, reps=50):
    """Trace `fn` and time the replay, so host dispatch is out of the number."""
    for _ in range(2):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    kept = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) * 1e6 / reps

    ttnn.release_trace(device, tid)
    ttnn.deallocate(kept)
    return us


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=134217728)
    try:
        print("\n  one batch-1 linear, traced; us per call and implied weight bandwidth")
        print(f"  {'shape':<16}{'grid':<12}{'us':>9}{'GB/s':>9}")
        total_default = 0.0
        best_total = 0.0
        for k, n in SHAPES:
            x = ttnn.from_torch(torch.randn(1, 1, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            w = ttnn.from_torch(torch.randn(k, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            mb = k * n * 2 / 1e6

            best = None
            for label, grid in [("default", None), ("8x8", (8, 8)), ("8x10", (8, 10)), ("4x8", (4, 8))]:
                try:
                    if grid is None:
                        us = timed(device, lambda: ttnn.linear(x, w))
                    else:
                        cg = ttnn.CoreGrid(y=grid[0], x=grid[1])
                        us = timed(device, lambda cg=cg: ttnn.linear(x, w, core_grid=cg))
                    print(f"  {f'{k}x{n}':<16}{label:<12}{us:>9.1f}{mb / us * 1e3:>9.1f}")
                    if label == "default":
                        total_default += us
                    best = us if best is None else min(best, us)
                except Exception as e:
                    print(f"  {f'{k}x{n}':<16}{label:<12}{'RAISED ' + type(e).__name__:>9}  {str(e)[:70]}")
            best_total += best or 0.0
            ttnn.deallocate(x)
            ttnn.deallocate(w)

        print(f"\n  per layer, default grid: {total_default:.1f} us   best grid: {best_total:.1f} us")
        print(f"  14 layers: {total_default * 14 / 1e3:.2f} ms  ->  {best_total * 14 / 1e3:.2f} ms")
        print("  Measured decode step is 8.25 ms, so the rest is the ~280 non-linear ops.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
