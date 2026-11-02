# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Can A1-stock's epilogue be made cheap enough to matter?

The stage breakdown says the matmul in A1-stock costs the same as stock
``ttnn.linear`` at M=1 (199 vs 208 us on gdn_qkvz) -- the G masked rows are
nearly free on a bandwidth-bound matmul. So A1-stock's whole deficit is the
epilogue: mask-mul (30 us) + scale-mul (45 us) + sum (31 us) = 112 us on tiny
tensors.

This measures the variants that could cut that, and checks each against a
float64 reference so a fast-but-wrong variant cannot win:

  base        fp32 scale, fp32 partials          45 + 31 us
  bf16 scale  bf16 scale, fp32 partials          half the scale traffic
  bf16 epi    bf16 scale, bf16 scale-mul output  half of both
  no mask     the matmul with no mask multiply, to price the mask step alone

If the best variant still leaves A1-stock at or above bfp8's time, the epilogue
has to move into the kernel (the plan's A3-fused) before the scheme can pay.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_epilogue
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .record import DRAM_PEAK_GBS
from .timing import Timer


def pcc(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    import ttnn

    from ..linear_stock import block_mask, padded_rows, scale_matrix

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    k, n, group, m = args.k, args.n, args.group, args.m
    rows = padded_rows(k // group)
    rng = np.random.default_rng(0)
    codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
    scales = rng.uniform(1e-3, 2e-2, size=(n, k // group)).astype(np.float16)
    x_np = rng.standard_normal((m, k)).astype(np.float32)
    sfull = np.repeat(scales.astype(np.float64), group, axis=1)
    expected = x_np.astype(np.float64) @ (codes.astype(np.float64) * sfull).T

    def dev(a, dt, mem=ttnn.DRAM_MEMORY_CONFIG):
        t = torch.from_numpy(np.ascontiguousarray(a))
        if dt == ttnn.bfloat16:
            t = t.to(torch.bfloat16)
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)

    DT = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16}
    try:
        w = dev(codes.astype(np.float32).T.copy(), ttnn.bfloat4_b)
        mask = ttnn.reshape(dev(block_mask(k, group), ttnn.bfloat16), (1, 1, rows, k))
        smat = scale_matrix(scales, group)
        xt = torch.from_numpy(x_np).to(torch.bfloat16)
        x_dev = ttnn.reshape(dev(x_np, ttnn.bfloat16), (m, 1, 1, k))

        # Build the masked activation once so the variants differ only in the epilogue.
        x_bd = ttnn.multiply(x_dev, mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        flat = ttnn.reshape(x_bd, (1, 1, m * rows, k))

        print(f"K={k} N={n} G={rows} M={m}, peak {DRAM_PEAK_GBS} GB/s")
        print(f"{'variant':20s} {'us':>9} {'GB/s':>7} {'pcc':>12}")

        def run(label, sdt, pdt):
            scale = ttnn.reshape(dev(smat, DT[sdt]), (1, 1, rows, n))

            def once():
                p = ttnn.linear(flat, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                p3 = ttnn.reshape(p, (m, 1, rows, n))
                sc = ttnn.multiply(p3, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   dtype=DT[pdt])
                y = ttnn.sum(sc, dim=2, keepdim=True)
                ttnn.synchronize_device(device)
                got = ttnn.to_torch(y).float().numpy().reshape(m, n)
                for t in (p, p3, sc, y):
                    ttnn.deallocate(t)
                return got

            med, _ = timer.best(lambda: once())
            got = once()
            c = pcc(got, expected)
            gbs = (n * k // 2 + n * k // 16 + rows * n * 4) / med / 1e9
            print(f"{label:20s} {med * 1e6:9.1f} {gbs:7.1f} {c:12.7f}"
                  f"{'' if c >= 0.9999 else '   <-- FAILS PCC'}", flush=True)

        run("base fp32/fp32", "fp32", "fp32")
        run("bf16 scale", "bf16", "fp32")
        run("bf16 scale+epi", "bf16", "bf16")

        # Price the mask step on its own.
        med, _ = timer.best(lambda: ttnn.deallocate(ttnn.multiply(x_dev, mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
        print(f"{'mask-mul alone':20s} {med * 1e6:9.1f}")

        # And the matmul with a plain (unmasked) activation, for the no-mask floor.
        plain = ttnn.reshape(dev(x_np, ttnn.bfloat16), (1, 1, m, k))
        med, _ = timer.best(lambda: ttnn.deallocate(ttnn.linear(plain, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
        print(f"{'linear M=1 (no mask)':20s} {med * 1e6:9.1f}")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
