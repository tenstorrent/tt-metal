# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where does A1-stock's time go? One stage at a time, on the device.

A1-stock measured 590 us on gdn_qkvz against 208 us for a plain bfp4 linear at
M=1, while the A0 fit predicted the block-masked scheme would *beat* bfp4 (104 us
at G=40). The gap has to be attributed before anything is tuned, so this times
every stage of the chain separately:

    mask-mul    [m,1,1,K] * [1,1,G,K]         -> [m,1,G,K]
    linear      [1,1,m*G,K] @ [K,N] bfp4      -> [1,1,m*G,N]
    reshape     [1,1,m*G,N] -> [m,1,G,N]      (metadata only?)
    scale-mul   [m,1,G,N] * [1,1,G,N] fp32    -> [m,1,G,N]
    sum         dim=2 of [m,1,G,N]            -> [m,1,1,N]
    readback    to_torch

plus the bytes each one moves, so a stage that is slow for a reason (it streams
the scale table) is distinguishable from one that is slow because of a bad
program config or a L1/DRAM placement mistake.

Both wall clock and the device-side kernel total are reported: the wall clock
includes host work, which for a chain of six ops can dominate at these sizes.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_stages --shape gdn_qkvz --m 1
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .record import DRAM_PEAK_GBS
from .timing import Timer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="gdn_qkvz")
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--m", default="1")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    import ttnn

    from ..linear_stock import block_mask, padded_rows, scale_matrix, TernaryLinearStock

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    k, n, group = args.k, args.n, args.group
    rows = padded_rows(k // group)
    rng = np.random.default_rng(0)
    codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
    scales = rng.uniform(1e-3, 2e-2, size=(n, k // group)).astype(np.float16)

    def report(label, med, nbytes, extra=""):
        gbs = nbytes / med / 1e9 if med > 0 else 0
        print(f"  {label:14s} {med * 1e6:8.1f} us  {nbytes / 1e6:7.2f} MB  "
              f"{gbs:7.1f} GB/s  {extra}", flush=True)

    try:
        for m in [int(x) for x in args.m.split(",")]:
            print(f"\n{args.shape} K={k} N={n} G={rows} M={m}", flush=True)
            eng = TernaryLinearStock(device, codes, scales, group)
            x_np = rng.standard_normal((m, k)).astype(np.float32)
            xt = torch.from_numpy(x_np)
            wt_bytes = eng.weight_bytes

            # ---- whole call, for the reference number
            def s_whole():
                y = eng(xt)
                ttnn.synchronize_device(device)
                ttnn.deallocate(y)

            med, _ = timer.best(s_whole)
            report("whole call", med, wt_bytes, "device tensor out, no readback")

            # ---- stage by stage
            mem = ttnn.DRAM_MEMORY_CONFIG
            t_x = ttnn.reshape(
                ttnn.from_torch(xt.to(torch.bfloat16).reshape(m, 1, 1, k), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem),
                (m, 1, 1, k),
            )

            def s_mask():
                y = ttnn.multiply(t_x, eng.mask, memory_config=mem)
                ttnn.synchronize_device(device)
                ttnn.deallocate(y)

            x_bd = ttnn.multiply(t_x, eng.mask, memory_config=mem)
            med, _ = timer.best(s_mask)
            report("mask-mul", med, rows * k * 2 + m * rows * k * 2, f"out {(m*rows*k*2)/1e6:.2f} MB")

            flat = ttnn.reshape(x_bd, (1, 1, m * rows, k))

            def s_linear():
                y = ttnn.linear(flat, eng.w, memory_config=mem)
                ttnn.synchronize_device(device)
                ttnn.deallocate(y)

            med, _ = timer.best(s_linear)
            report("linear", med, wt_bytes, f"[1,1,{m*rows},{k}] @ [{k},{n}] bfp4")

            # The same linear with the activation in L1 instead of DRAM.
            def s_linear_l1():
                y = ttnn.linear(flat, eng.w, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.synchronize_device(device)
                ttnn.deallocate(y)

            try:
                med, _ = timer.best(s_linear_l1)
                report("linear(L1 out)", med, wt_bytes, "")
            except Exception as e:  # noqa: BLE001
                print(f"  linear(L1 out)  FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)

            p = ttnn.linear(flat, eng.w, memory_config=mem)
            p3 = ttnn.reshape(p, (m, 1, rows, n))

            def s_reshape():
                y = ttnn.reshape(p, (m, 1, rows, n))
                ttnn.synchronize_device(device)

            med, _ = timer.best(s_reshape)
            report("reshape", med, 0, "metadata")

            def s_scale():
                y = ttnn.multiply(p3, eng.scale, memory_config=mem)
                ttnn.synchronize_device(device)
                ttnn.deallocate(y)

            med, _ = timer.best(s_scale)
            report("scale-mul", med, m * rows * n * 2 + rows * n * 4,
                   f"reads {m*rows*n*2/1e6:.2f} MB bf16 + {rows*n*4/1e6:.2f} MB fp32 scale")

            scaled = ttnn.multiply(p3, eng.scale, memory_config=mem)

            def s_sum():
                y = ttnn.sum(scaled, dim=2, keepdim=True)
                ttnn.synchronize_device(device)
                ttnn.deallocate(y)

            med, _ = timer.best(s_sum)
            report("sum dim=2", med, m * rows * n * 4, "fp32 read")

            # Scale fold in bf16 instead of fp32: half the scale traffic, and the
            # partials are bf16 anyway so fp32 may buy nothing but bandwidth.
            med, _ = timer.best(lambda: (ttnn.to_torch(ttnn.sum(scaled, dim=2, keepdim=True))))
            report("sum+readback", med, 0, "")

            print(f"  (A0 reference: bfp4 linear M=1 on this shape was 193 us, "
                  f"M={rows} was measured in a0_stock_regime)", flush=True)

            for t in (t_x, x_bd, flat, p, p3, scaled, eng.w, eng.mask, eng.scale):
                try:
                    ttnn.deallocate(t)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
