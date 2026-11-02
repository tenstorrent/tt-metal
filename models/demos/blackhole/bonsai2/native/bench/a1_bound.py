# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is the decode matmul bandwidth-bound or compute-bound? This decides the engine.

PLAN.md's A1-stock projection assumes bytes buy time roughly linearly: it fits
stock bfp4 at a shape and predicts what the block-masked scheme would cost,
concluding 1.9x over today's bfp8. The A0-stock sweep already showed one crack in
that -- ffn_down takes the same 376 us at bfp4 as at bfp8 despite half the bytes
-- and the A1-stock measurements show a second: the block-masked matmul lands at
the M=1 time (extra rows are nearly free), so A1-stock can never beat a plain
bfp4 matmul because it reads the same bytes and does more work.

The question that decides everything is therefore not "how many bytes" but "how
much of the time is compute". Format does not change the MVMUL work: a bfp4 tile
and a bf16 tile are the same 32x32 tile, unpacked and multiplied the same way.
Only the DRAM traffic differs. So at fixed [K, N]:

    t(fmt) = compute + bytes(fmt) / marginal_BW

Three formats give the line, and its intercept is the compute floor -- the
fastest any format can ever run that shape. If the intercept is a large share of
today's time, no amount of byte reduction (bfp2, forced exponent, or PTQ1_0's
1.5 bits) can reach the plan's rooflines on that shape, and the plan's per-shape
tok/s projections need revising rather than tuning.

Also checks whether an explicit matmul config lifts bfp4's achieved bandwidth,
because a 44%-of-peak number that is really a bad config would change the
conclusion.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_bound
"""

from __future__ import annotations

import argparse
import os

import torch

from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer

#: (name, K, N). The decode-critical shapes, plus ffn_down as the suspected
#: compute-bound case.
SHAPES = [
    ("gdn_qkvz", 5120, 16384),
    ("attn_q_gate", 5120, 12288),
    ("ffn_gate", 5120, 17408),
    ("ffn_down", 17408, 5120),
    ("lm_head", 5120, 248320),
]


def weight_bytes(n: int, k: int, fmt: str) -> int:
    """On-device weight bytes including BFP block overhead."""
    if fmt == "bf16":
        return n * k * 2
    if fmt == "bfp8":
        return n * k + n * k // 16
    if fmt == "bfp4":
        return n * k // 2 + n * k // 16
    raise ValueError(fmt)


def fit(xs, ys):
    """Least squares y = a + b x; returns (a, b) and the residual max."""
    nn = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = nn * sxx - sx * sx
    if not den:
        return 0.0, 0.0, 0.0
    b = (nn * sxy - sx * sy) / den
    a = (sy - b * sx) / nn
    res = max(abs(y - (a + b * x)) for x, y in zip(xs, ys))
    return a, b, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--formats", default="bf16,bfp8,bfp4")
    ap.add_argument("--m", default="1")
    ap.add_argument("--xin", default="dram", help="activation memory config: dram (model-like) or l1")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a1_bound.jsonl"))
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
    formats = args.formats.split(",")

    print(f"Peak {DRAM_PEAK_GBS} GB/s.  Fit per (shape, M): t = compute + bytes/BW.  x in {args.xin.upper()}")
    print(f"{'shape':12s} {'M':>3} " + " ".join(f"{f:>9s}" for f in formats))
    try:
        for name, k, n in shapes:
            w_src = torch.randn((k, n), dtype=torch.bfloat16)
            for m in ms:
                times, byts = {}, {}
                x_mem = ttnn.L1_MEMORY_CONFIG if args.xin == "l1" else ttnn.DRAM_MEMORY_CONFIG
                x = ttnn.from_torch(
                    torch.randn((1, 1, m, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device, memory_config=x_mem,
                )
                for fmt in formats:
                    try:
                        wt = ttnn.from_torch(
                            w_src, dtype=dt[fmt], layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"  {name} {fmt}: alloc FAILED {str(e)[:90]}", flush=True)
                        continue

                    def once():
                        y = ttnn.linear(x, wt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.synchronize_device(device)
                        ttnn.deallocate(y)

                    try:
                        med, samples = timer.best(once)
                    except Exception as e:  # noqa: BLE001
                        print(f"  {name} {fmt}: FAILED {str(e)[:90]}", flush=True)
                        ttnn.deallocate(wt)
                        continue
                    times[fmt] = med * 1e6
                    byts[fmt] = weight_bytes(n, k, fmt) / 1e6
                    rec = new_record(f"a1bound_{name}", K=k, N=n, M=m, fmt=fmt)
                    rec.samples = samples
                    append_jsonl(args.out, rec, int(byts[fmt] * 1e6))
                    ttnn.deallocate(wt)

                cells = " ".join(f"{times[f]:9.1f}" if f in times else f"{'-':>9s}" for f in formats)
                print(f"{name:12s} {m:3d} {cells}", flush=True)

                have = [f for f in formats if f in times]
                if len(have) >= 2:
                    a, b, res = fit([byts[f] for f in have], [times[f] for f in have])
                    print(f"{'':12s} {'':3s} compute floor ~{a:6.1f} us, marginal "
                          f"{b / 1e3:6.1f} us/MB -> {1e3 / b:6.1f} GB/s   (fit residual {res:.1f} us)")
                    if a > 0:
                        # What the plan's rooflines would imply if only format mattered.
                        for target_name, tbits in (("bfp2", 2.125), ("forced-exp bfp2", 1.9)):
                            tb = n * k * tbits / 8 / 1e6
                            floor = a + b * tb
                            print(f"{'':15s} {target_name:16s} {tb:7.2f} MB -> {floor:6.1f} us "
                                  f"({times.get('bfp4', floor) / floor:.2f}x vs bfp4)")
                ttnn.deallocate(x)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
