# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Is the KV-cache shift expensive because of bytes, or because of tile alignment?

Per-op pricing put `slice` + `concat` on the `[1, 16, 256, 64]` cache at ~228 us a
layer -- 0.5 MB moved in 134 us is about 3.7 GB/s, two orders below what a copy of
that size should cost. So the cost is probably not the bytes.

The hypothesis is layout. In `TILE_LAYOUT` rows live in 32-row tiles, and both halves
of the shift are misaligned to that: slicing from row 1 and concatenating a 1-row
tensor onto a 255-row one each require re-tiling the whole buffer by one row. If that
is right, the same operations at a 32-row granularity should be far cheaper, and the
fix is to shift a tile at a time rather than a row at a time.

This measures the same ops at row and tile granularity, and `bfloat8_b` alongside, to
separate layout from bytes.

    python models/demos/cosyvoice/scripts/probe_kv_alignment.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

H, DK, MAX_LEN = 16, 64, 256
TILE = 32
REPS = 40
CHAIN = 16  # ops per trace, so the ~12 us replay launch is amortised rather than counted


def timed(device, make_op):
    """Time a chain of `CHAIN` independent ops in one trace; return us per op."""
    for _ in range(2):
        for _ in range(CHAIN):
            out = make_op()
            if out is not None:
                ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    kept = [make_op() for _ in range(CHAIN)]
    ttnn.end_trace_capture(device, tid, cq_id=0)

    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) * 1e6 / REPS / CHAIN

    ttnn.release_trace(device, tid)
    for k in kept:
        if k is not None:
            ttnn.deallocate(k)
    return us


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)
    try:
        # bfloat8_b separates layout cost from byte cost; L1 separates it from DRAM
        # latency. If the misaligned shift is re-tiling compute, neither should move it.
        for dt, mc, name in (
            (ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, "bfloat16 / DRAM"),
            (ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG, "bfloat8_b / DRAM"),
            (ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, "bfloat16 / L1"),
        ):
            mk = lambda x: ttnn.from_torch(  # noqa: E731
                x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
            )
            buf = mk(torch.randn(1, H, MAX_LEN, DK))
            one = mk(torch.randn(1, H, 1, DK))
            tile = mk(torch.randn(1, H, TILE, DK))

            print(
                f"\n  {name}: cache [1, {H}, {MAX_LEN}, {DK}]  "
                f"({H * MAX_LEN * DK * (2 if dt == ttnn.bfloat16 else 1) / 1e6:.2f} MB)"
            )
            print(f"  {'operation':<44}{'us':>9}")

            rows = [
                ("slice from row 1 (current shift)", lambda: ttnn.slice(buf, [0, 0, 1, 0], [1, H, MAX_LEN, DK])),
                ("slice from row 32 (tile-aligned)", lambda: ttnn.slice(buf, [0, 0, TILE, 0], [1, H, MAX_LEN, DK])),
                ("slice from row 0 (no-op offset)", lambda: ttnn.slice(buf, [0, 0, 0, 0], [1, H, MAX_LEN - 1, DK])),
                (
                    "concat 255 + 1 row (current append)",
                    lambda: ttnn.concat([ttnn.slice(buf, [0, 0, 1, 0], [1, H, MAX_LEN, DK]), one], dim=2),
                ),
                (
                    "concat 224 + 32 rows (tile-aligned)",
                    lambda: ttnn.concat([ttnn.slice(buf, [0, 0, TILE, 0], [1, H, MAX_LEN, DK]), tile], dim=2),
                ),
                ("copy whole buffer", lambda: ttnn.copy(buf, buf)),
            ]
            for label, fn in rows:
                try:
                    print(f"  {label:<44}{timed(device, fn):>9.1f}")
                except Exception as e:  # noqa: BLE001
                    print(f"  {label:<44}{'RAISED':>9}  {type(e).__name__}: {str(e)[:60]}")

            for t in (buf, one, tile):
                ttnn.deallocate(t)

        print("\n  If tile-aligned is much cheaper, the shift should move 32 rows every 32")
        print("  steps rather than 1 row every step, and the win is layout, not bytes.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
