# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is stock ``ttnn.linear`` bandwidth-bound, and at what shape does it stop being?

A1-stock trades DRAM traffic for MVMUL work: it multiplies the token count by
``G = K/group`` so each group's partial sum comes out of one matmul. That only
pays if the matmul is bandwidth-bound, because the weight is read once either
way. The A0-stock sweep already found one case where it is not: at
``ffn_down`` (K=17408, N=5120) bfp4 takes 375 us, exactly the same as bfp8
despite half the bytes, so that shape is compute-bound and A1-stock's G rows
would multiply a cost that is already compute.

This finds the boundary instead of guessing it. For each shape it fits

    time = max(bytes / BW, M * work_per_row)

by measuring the same shape at several M and several formats, then reports
which term dominates and how much headroom M has before compute takes over.
That headroom is exactly what A1-stock can spend: M=G is affordable only when
``compute_at_G < bytes / BW``.

Measured at M = 1, 32, 64, 128 so the compute slope is identifiable.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a0_stock_regime
"""

from __future__ import annotations

import argparse
import os

import torch

from .a0_stock import SHAPES, weight_bytes
from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="ffn_gate,ffn_down,gdn_qkvz")
    ap.add_argument("--formats", default="bfp8,bfp4")
    ap.add_argument("--m", default="1,32,64,128")
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a0_stock_regime.jsonl"))
    args = ap.parse_args()

    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    shapes = SHAPES
    if args.shapes:
        want = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in want]
    ms = [int(x) for x in args.m.split(",")]

    print(f"DRAM-sharded (decode) config, activation in L1.  Peak {DRAM_PEAK_GBS} GB/s")
    print(f"{'shape':12s} {'fmt':5s} {'MB':>7} " + " ".join(f"M={m:<7d}" for m in ms) + "  us/row (fit)")
    try:
        for name, k, n in shapes:
            w_src = torch.randn((k, n), dtype=torch.bfloat16)
            for fmt in args.formats.split(","):
                wt = ttnn.from_torch(
                    w_src, dtype=dt[fmt], layout=ttnn.TILE_LAYOUT,
                    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                nb = weight_bytes(n, k, fmt)
                times = {}
                for m in ms:
                    x = ttnn.from_torch(
                        torch.randn((1, 1, m, k), dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
                    )

                    def once():
                        y = ttnn.linear(x, wt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.synchronize_device(device)
                        ttnn.deallocate(y)

                    try:
                        med, samples = timer.best(once)
                        times[m] = med
                        rec = new_record(f"a0regime_{name}", K=k, N=n, M=m, fmt=fmt)
                        rec.samples = samples
                        append_jsonl(args.out, rec, nb)
                    except Exception as e:  # noqa: BLE001
                        print(f"  {name} {fmt} M={m}: FAILED {type(e).__name__}: {e}", flush=True)
                    finally:
                        ttnn.deallocate(x)

                if len(times) < 2:
                    continue
                # Least-squares slope of time against M; the intercept is the
                # fixed cost and the slope the marginal cost per extra row.
                xs = sorted(times)
                ys = [times[m] * 1e6 for m in xs]
                nn = len(xs)
                sx, sy = sum(xs), sum(ys)
                sxx = sum(x * x for x in xs)
                sxy = sum(x * y for x, y in zip(xs, ys))
                den = nn * sxx - sx * sx
                slope = (nn * sxy - sx * sy) / den if den else 0.0
                inter = (sy - slope * sx) / nn
                bandwidth_us = nb / (DRAM_PEAK_GBS * 1e9) * 1e6
                cells = " ".join(f"{times[m] * 1e6:7.1f}" for m in xs if m in times)
                print(
                    f"{name:12s} {fmt:5s} {nb / 1e6:7.2f} {cells}  {slope:7.2f}"
                    f"   (intercept {inter:6.1f} us, ideal {bandwidth_us:6.1f} us)"
                )
                ttnn.deallocate(wt)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
