# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 35: K-block depth of the 2D (block-mcast) matmuls against ttnn's own choice on the same grid (K blocks
of 4), HiFi4 / fp32 accumulate / packer L1 accumulate; outputs compared to ttnn's choice. Default: the dX^T form
(second operand [N, K], transpose_b) on the full grid; ``--no-tb`` for the forward form, ``--grid XxY`` for a
smaller grid (the w2 forward's 11x8).

    python iter35_dx_depth.py --m 5184 --k 16384 --n 1024 --depths 4,8,16
    python iter35_dx_depth.py --m 1024 --k 8320 --n 5120 --no-tb --grid 11x8 --depths 6,10,13,20 --subs 4x1,2x2
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import ttnn


def bench(fn, reps, device):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e3 / reps, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--depths", default="4,8,16")
    p.add_argument("--subs", default="1x3,2x1,1x2,2x2")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--no-tb", action="store_true", help="forward form: the second operand is [K, N]")
    p.add_argument("--grid", default=None, help="XxY grid instead of the full one")
    args = p.parse_args()
    tb = not args.no_tb
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    if args.grid:
        gx, gy = (int(v) for v in args.grid.split("x"))
    else:
        gx, gy = grid.x, grid.y
    M, K, N = args.m, args.k, args.n
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )
    rng = np.random.default_rng(0)
    a = ttnn.from_torch(
        torch.from_numpy(rng.standard_normal((1, 1, M, K)).astype(np.float32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    b = ttnn.from_torch(
        torch.from_numpy((rng.standard_normal((1, 1, N, K) if tb else (1, 1, K, N)) * 0.02).astype(np.float32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    flops = 2 * M * K * N
    ms, ref = bench(
        lambda: ttnn.matmul(a, b, transpose_b=tb, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=gy, x=gx)),
        args.reps,
        device,
    )
    ref_h = ttnn.to_torch(ref).float()
    print(
        f"M={M} K={K} N={N}{' transpose_b' if tb else ''}: ttnn's choice on {gx}x{gy}: {ms:.3f} ms {flops / ms / 1e9:.1f} TFLOP/s"
    )
    pcM, pcN = -(-Mt // gy), -(-Nt // gx)
    for d in (int(v) for v in args.depths.split(",")):
        if Kt % d:
            print(f"  depth {d}: Kt {Kt} not divisible")
            continue
        for sub in args.subs.split(","):
            sh, sw = (int(v) for v in sub.split("x"))
            if pcM % sh or pcN % sw or sh * sw > 4:
                continue
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=d,
                out_subblock_h=sh,
                out_subblock_w=sw,
                per_core_M=pcM,
                per_core_N=pcN,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
            try:
                t, out = bench(
                    lambda: ttnn.matmul(a, b, transpose_b=tb, program_config=pc, compute_kernel_config=cfg),
                    args.reps,
                    device,
                )
                err = float((ttnn.to_torch(out).float() - ref_h).abs().max())
                ttnn.deallocate(out)
                print(
                    f"  2D {gx}x{gy} pcM {pcM} pcN {pcN} depth {d:2d} sub {sh}x{sw}: {t:.3f} ms {flops / t / 1e9:6.1f} TFLOP/s  max|diff| {err:.2e}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  2D depth {d} sub {sh}x{sw}: FAILED {str(e).splitlines()[0][:100]}", flush=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
