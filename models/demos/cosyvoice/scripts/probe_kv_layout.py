# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Can the KV shift be made cheap by moving time onto a free axis?

`TILE_LAYOUT` tiles only the **last two** dimensions. The cache is `[1, h, T, dk]`, so
time is a tiled axis, and that is the whole reason slicing from row 1 costs 78 us and
appending one row costs 207 -- both re-tile the buffer (see `probe_kv_alignment.py`).

Dims 0 and 1 are not tiled. If the cache were laid out with time there, the shift should
be a cheap strided copy, at the price of one `permute` per layer to get back to
`[b, h, T, dk]` for `q @ k^T`. That trade is worth taking only if

    slice + concat + permute   <<   78 + 207

and if it is, it captures most of what `update_cache` offers (F41) **without** the
32-trace rework that design needs: no baked write index, no per-sub-step positional
offset, no change to the attention geometry at all.

The catch to watch for is padding. `[1, T, h, dk]` tiles `(h, dk) = (16, 64)`, and 16
pads to 32 -- so the buffer doubles. That is 1 MB per tensor per layer, 28 MB total,
which is affordable if the timing works out.

    python models/demos/cosyvoice/scripts/probe_kv_layout.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

H, DK, T = 16, 64, 256
REPS = 40
CHAIN = 16


def timed(device, make):
    """Per-op cost inside a trace, with the ~12 us replay launch amortised over CHAIN."""
    for _ in range(2):
        for _ in range(CHAIN):
            out = make()
            if out is not None:
                ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    kept = [make() for _ in range(CHAIN)]
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
        mk = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731

        print(f"\n  {'layout / operation':<46}{'us':>9}")
        print("  " + "-" * 55)

        # --- the shipped layout: time on a TILED axis ---
        cur = mk(torch.randn(1, H, T, DK))
        one_cur = mk(torch.randn(1, H, 1, DK))
        a = timed(device, lambda: ttnn.slice(cur, [0, 0, 1, 0], [1, H, T, DK]))
        b = timed(
            device,
            lambda: ttnn.concat([ttnn.slice(cur, [0, 0, 1, 0], [1, H, T, DK]), one_cur], dim=2),
        )
        print(f"  {'[1,h,T,dk]  slice row 1 (shipped)':<46}{a:>9.1f}")
        print(f"  {'[1,h,T,dk]  slice+concat (shipped)':<46}{b:>9.1f}")
        shipped = b + 0.0

        # --- time on dim 1, a FREE axis ---
        alt = mk(torch.randn(1, T, H, DK))
        one_alt = mk(torch.randn(1, 1, H, DK))
        try:
            c = timed(device, lambda: ttnn.slice(alt, [0, 1, 0, 0], [1, T, H, DK]))
            d = timed(
                device,
                lambda: ttnn.concat([ttnn.slice(alt, [0, 1, 0, 0], [1, T, H, DK]), one_alt], dim=1),
            )
            print(f"  {'[1,T,h,dk]  slice on dim 1 (free axis)':<46}{c:>9.1f}")
            print(f"  {'[1,T,h,dk]  slice+concat on dim 1':<46}{d:>9.1f}")
        except Exception as e:  # noqa: BLE001
            d = None
            print(f"  {'[1,T,h,dk]  dim-1 shift':<46}{'RAISED':>9}  {str(e)[:60]}")

        # --- the permute that layout would cost, per layer per step ---
        try:
            p = timed(device, lambda: ttnn.permute(alt, (0, 2, 1, 3)))
            print(f"  {'[1,T,h,dk] -> [1,h,T,dk]  permute':<46}{p:>9.1f}")
        except Exception as e:  # noqa: BLE001
            p = None
            print(f"  {'permute (0,2,1,3)':<46}{'RAISED':>9}  {str(e)[:60]}")

        # --- time on dim 0 instead, in case dim 1 is special ---
        alt0 = mk(torch.randn(T, 1, H, DK))
        one0 = mk(torch.randn(1, 1, H, DK))
        try:
            e0 = timed(
                device,
                lambda: ttnn.concat([ttnn.slice(alt0, [1, 0, 0, 0], [T, 1, H, DK]), one0], dim=0),
            )
            print(f"  {'[T,1,h,dk]  slice+concat on dim 0':<46}{e0:>9.1f}")
        except Exception as ex:  # noqa: BLE001
            print(f"  {'[T,1,h,dk]  dim-0 shift':<46}{'RAISED':>9}  {str(ex)[:60]}")

        if d is not None and p is not None:
            print(f"\n  per tensor per layer:  shipped {shipped:.1f} us   ->   free-axis {d + p:.1f} us")
            print(f"  ratio {shipped / (d + p):.2f}x   (k and v both, x14 layers)")
            saved = 2 * 14 * (shipped - (d + p)) / 1e3
            print(f"  would remove ~{saved:.2f} ms from an 8.23 ms step, if it holds in situ")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
