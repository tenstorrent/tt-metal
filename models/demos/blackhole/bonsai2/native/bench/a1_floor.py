# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is the matmul's fixed cost a launch cost or real work? It decides the rooflines.

Fitting t = compute + bytes/BW over bf16/bfp8/bfp4 at M=1 gives an intercept of
78 us on gdn_qkvz, 82 us on ffn_gate, 315 us on ffn_down, 996 us on lm_head.
If that intercept is a *fixed* cost (launch, setup, the activation broadcast) it
can be amortised away and the plan's byte-scaled rooflines are reachable. If it
scales with the number of weight tiles, it is real work that no format change
touches, and it caps what bfp2 can deliver.

Three sweeps separate the possibilities:

  K sweep, N=32      one output tile column: time vs K is pure streaming plus
                     whatever per-K-tile work exists. If time at K=5120 is
                     already tens of us, the cost is per-K-tile, not launch.
  N sweep, K=5120    more output tiles at fixed K. A linear rise means the cost
                     is per output tile; flat-then-rising means a threshold.
  M sweep, fixed     the same shape at M=1..64. A floor that is flat in M is not
                     MVMUL work (MVMUL is per tile and does not care about rows).

The M sweep is the decisive one for the plan: A1-stock spends G=40..136 rows per
projection on the assumption that extra rows are nearly free, and A0 already
showed the floor is flat to M=32. If it stays flat to M=64 while the byte-scaled
term saturates, the format lever is real but the per-tile work is not.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_floor
"""

from __future__ import annotations

import argparse
import os

import torch

from .record import DRAM_PEAK_GBS
from .timing import Timer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}

    def bench(k, n, m, fmt):
        w = ttnn.from_torch(
            torch.randn((k, n), dtype=torch.bfloat16), dtype=dt[fmt], layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.from_torch(
            torch.randn((1, 1, m, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def once():
            y = ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.synchronize_device(device)
            ttnn.deallocate(y)

        try:
            med, _ = timer.best(once)
        finally:
            ttnn.deallocate(w)
            ttnn.deallocate(x)
        return med * 1e6

    print(f"Peak {DRAM_PEAK_GBS} GB/s, M=1, x in DRAM, 5-run medians")
    try:
        print("\nK sweep at N=32 (one output tile column), bfp8:")
        for k in (512, 1024, 2048, 5120, 10240, 17408):
            us = bench(k, 32, 1, "bfp8")
            mb = (k * 32 + k * 32 // 16) / 1e6
            print(f"  K={k:6d} N=32   {us:8.1f} us   {mb:6.2f} MB   {mb / us * 1e3:6.1f} GB/s",
                  flush=True)

        print("\nN sweep at K=5120, bfp8 (tiles = (K/32)*(N/32)):")
        for n in (32, 128, 512, 2048, 8192, 16384, 32768):
            us = bench(5120, n, 1, "bfp8")
            mb = (5120 * n + 5120 * n // 16) / 1e6
            tiles = (5120 // 32) * (n // 32)
            print(f"  N={n:6d}  {us:8.1f} us   {mb:7.2f} MB   {tiles:7d} tiles   "
                  f"{mb / us * 1e3:6.1f} GB/s   {us * 1e3 / tiles:5.1f} ns/tile", flush=True)

        print("\nM sweep at the decode shapes, bfp4 (a floor flat in M is not MVMUL):")
        for name, k, n in (("gdn_qkvz", 5120, 16384), ("ffn_down", 17408, 5120),
                           ("ffn_gate", 5120, 17408)):
            cells = []
            for m in (1, 2, 4, 8, 16, 32, 64):
                cells.append((m, bench(k, n, m, "bfp4")))
            print(f"  {name:10s} " + "  ".join(f"M={m}:{us:6.1f}" for m, us in cells), flush=True)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
