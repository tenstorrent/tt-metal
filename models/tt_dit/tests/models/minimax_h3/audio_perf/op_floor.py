"""What does one ttnn op cost before it moves any data, and what bandwidth does it reach?

RETRACTED 2026-08-12 -- do not quote this script's per-op number as a device floor. It synchronizes
between ops, so what it measures is host *issue* cost. `op_pipeline.py` shows chained 141.9 /
independent 125.7 / per-op-sync 138.8 us/op, all equal, i.e. that microbenchmark is host-issue-bound and
cannot see device time at all. The "6955 ops x 180 us = 1254 ms floor" derived from this is void; real
per-op device cost is ~37 us. Worse, the premise below ("Trace measured 1.00x, so it is not host
dispatch") is true only on a single device -- trace is 3.06x on a sharded 32-chip mesh, and taking that
1.00x unqualified is what sent a week of work down the kernel path. See ITEM1_RESULT.md and
ITEM2_RESULT.md.

Two results so far disagree about what binds this stage. Trace measured 1.00x, so it is not host
dispatch. bf16 measured 1.23x rather than 2x, so it is not purely bytes either. The remaining
candidate is fixed per-op device cost, and that is worth pinning down: a decode issues 6955 ops, so a
floor of F microseconds per op puts a hard 6955*F on the runtime no matter how good the kernels are.

Times a bandwidth-bound elementwise op across four decades of tensor size and fits

    ms(bytes) = fixed_overhead + bytes / bandwidth

The intercept is the per-op floor. The slope is the bandwidth actually achieved. Together they say
whether the path to 60 ms is fewer ops, fewer bytes, or both -- and what op count 60 ms even allows.
"""

import os
import statistics
import time

import torch

import ttnn

# Row counts spanning tiny to larger than anything in the decode, at the tail's channel width.
ROWS = [32, 256, 2048, 16384, 131072, 331212]
C = 8
ITERS = int(os.environ.get("FLOOR_ITERS", "10"))


def timed(fn, device, iters=ITERS):
    fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(iters):
        s = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - s) * 1e3)
    return statistics.median(ts)


def fit(xs, ys):
    """Least squares y = a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return my - b * mx, b


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        for dtype, name, esz in ((ttnn.float32, "float32", 4), (ttnn.bfloat16, "bfloat16", 2)):
            print(f"\n=== {name} elementwise add, B=2, C={C} ===")
            print(f"{'rows':>8} {'MB moved':>9} {'ms':>8} {'GB/s':>8}")
            xs, ys = [], []
            for rows in ROWS:
                x = torch.randn(2, rows, C) * 0.3
                xd = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                try:
                    ms = timed(lambda: ttnn.add(xd, xd), device)
                except Exception as exc:  # noqa: BLE001
                    print(f"{rows:>8} FAILED {str(exc).splitlines()[0][:50]}")
                    continue
                # two reads and one write
                mb = 3 * 2 * rows * C * esz / 1e6
                gbs = mb / 1e3 / (ms / 1e3)
                print(f"{rows:>8} {mb:>9.2f} {ms:>8.3f} {gbs:>8.1f}")
                xs.append(mb)
                ys.append(ms)
            if len(xs) >= 2:
                a, b = fit(xs, ys)
                bw = (1 / b) / 1e3 if b > 0 else float("inf")
                print(f"  fit: fixed {a * 1e3:.1f} us/op, slope -> {bw:.1f} GB/s")
                if a > 0:
                    print(f"  => 6955 ops x {a * 1e3:.1f} us = {6955 * a:.0f} ms floor at today's op count")
                    print(f"  => 60 ms budget allows ~{int(60 / a)} ops")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
