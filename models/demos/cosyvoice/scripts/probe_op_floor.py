# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""What does one TTNN op cost inside a trace, at decode-step tensor sizes?

The AR decode step is a chain of a few hundred ops on one-row tensors, and every
optimisation so far has been "issue fewer ops". That strategy has a floor, and the
floor is what decides whether the remaining RTF targets are reachable by fusion at
all or need a different kind of change. Measuring it converts "we did not reach
0.5" into "0.5 is N ops away and the floor is M us", which is a statement someone
can act on.

Trace replay is the right setting: it removes host dispatch, so what is left is
whatever the device charges per program, and that is the quantity in question.

    python models/demos/cosyvoice/scripts/probe_op_floor.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def bench(device, n_ops, shape, reps=20):
    """Time a traced chain of `n_ops` elementwise adds on `shape`."""
    a = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def body():
        acc = ttnn.add(a, b)
        for _ in range(n_ops - 1):
            nxt = ttnn.add(acc, b)
            ttnn.deallocate(acc)
            acc = nxt
        return acc

    for _ in range(2):  # warm up the program cache before capture
        out = body()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    kept = body()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) * 1e3 / reps

    ttnn.release_trace(device, tid)
    for t in (a, b, kept):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return ms


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=134217728)
    try:
        print("\n  traced chain of elementwise adds -- per-op cost by tensor size")
        print(f"  {'shape':<22}{'ops':>6}{'total ms':>11}{'us/op':>10}")
        for shape in [(1, 1, 1024), (1, 16, 1, 64), (1, 1, 4096), (2, 608, 512)]:
            for n in (32, 128):
                ms = bench(device, n, shape)
                print(f"  {str(shape):<22}{n:>6}{ms:>11.3f}{ms * 1e3 / n:>10.2f}")

        print("\n  A 14-layer decode step issues ~330 ops. At the floor above that is")
        print("  the irreducible cost of the current op decomposition.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
