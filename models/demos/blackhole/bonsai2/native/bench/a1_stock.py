# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A1-stock per-projection timing, next to the A0-stock roofline.

The plan's ask: "device us per projection next to your A0 table". The comparison
that matters is ternary-as-bfloat4_b (A1-stock, about 4.6 bits per weight) against
the same shape at bfp4 (which is the closest stock has to ternary, 4 bits) and
bfp8 (what most of the model uses today).

Three things get reported for each shape:

  a1        TernaryLinearStock, the real call: mask, linear, scale fold, reduce
  bfp4      stock ``ttnn.linear`` at M=1 with a bfloat4_b weight
  bfp8      stock ``ttnn.linear`` at M=1 with a bfloat8_b weight

A1-stock is measured for wall time and reported with its own byte count (bfp4
codes plus the scale table), so the GB/s column is the streaming rate it
actually achieves. The gap between a1 us and bfp4 us is the price of the G rows
and the extra ops; the gap between bfp4 us and bfp8 us is what the bytes buy.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_stock
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .a0_stock import SHAPES, weight_bytes
from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer

GROUP = 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--formats", default="a1,bfp4,bfp8")
    ap.add_argument("--m", default="1")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a1_stock.jsonl"))
    args = ap.parse_args()

    import ttnn

    from ..linear_stock import TernaryLinearStock

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    dt = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    shapes = SHAPES
    if args.shapes:
        want = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in want]
    ms = [int(x) for x in args.m.split(",")]
    formats = args.formats.split(",")

    print(f"Peak {DRAM_PEAK_GBS} GB/s.  a1 = block-masked scale fold (real ternary), "
          f"bfp4/bfp8 = stock linear at M=1.")
    print(f"{'shape':12s} {'M':>3} {'fmt':5s} {'MB':>7} {'us':>9} {'GB/s':>7} {'%peak':>6}  vs bfp4")
    try:
        for name, k, n in shapes:
            rng = np.random.default_rng(0)
            # Real ternary codes, not random floats: the codes are what the bfp4
            # weight holds, so the measurement is on the shapes the model runs.
            codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
            scales = rng.uniform(1e-3, 2e-2, size=(n, k // GROUP)).astype(np.float16)
            w_src = torch.from_numpy(codes.astype(np.float32).T.copy())
            base = {}
            for m in ms:
                x_np = rng.standard_normal((m, k)).astype(np.float32)
                for fmt in formats:
                    if fmt == "a1":
                        eng = TernaryLinearStock(device, codes, scales, GROUP)
                        xt = torch.from_numpy(x_np)

                        def once():
                            y = eng(xt)
                            ttnn.synchronize_device(device)
                            ttnn.deallocate(y)

                        nb = eng.weight_bytes
                    else:
                        wt = ttnn.from_torch(
                            w_src, dtype=dt[fmt], layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        x = ttnn.from_torch(
                            torch.from_numpy(x_np).to(torch.bfloat16).reshape(1, 1, m, k),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
                        )

                        def once():
                            y = ttnn.linear(x, wt, memory_config=ttnn.L1_MEMORY_CONFIG)
                            ttnn.synchronize_device(device)
                            ttnn.deallocate(y)

                        nb = weight_bytes(n, k, fmt)
                    try:
                        med, samples = timer.best(once)
                    except Exception as e:  # noqa: BLE001
                        print(f"{name:12s} {m:3d} {fmt:5s}: FAILED {type(e).__name__}: {str(e)[:120]}",
                              flush=True)
                        if fmt != "a1":
                            ttnn.deallocate(wt)
                            ttnn.deallocate(x)
                        else:
                            ttnn.deallocate(eng.w)
                            ttnn.deallocate(eng.mask)
                            ttnn.deallocate(eng.scale)
                        continue
                    rec = new_record(f"a1stock_{name}", K=k, N=n, M=m, fmt=fmt, group=GROUP)
                    rec.samples = samples
                    append_jsonl(args.out, rec, nb)
                    gbs = nb / med / 1e9
                    if fmt == "bfp4":
                        base[m] = med
                    ratio = f"{(med / base[m]):.2f}x" if m in base and fmt == "a1" else ""
                    print(f"{name:12s} {m:3d} {fmt:5s} {nb / 1e6:7.2f} {med * 1e6:9.1f} "
                          f"{gbs:7.1f} {gbs / DRAM_PEAK_GBS * 100:5.1f}%  {ratio}", flush=True)
                    if fmt == "a1":
                        ttnn.deallocate(eng.w)
                        ttnn.deallocate(eng.mask)
                        ttnn.deallocate(eng.scale)
                    else:
                        ttnn.deallocate(wt)
                        ttnn.deallocate(x)
            print("", flush=True)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
