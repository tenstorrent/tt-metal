# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Program-config sweep for one matmul shape (HiFi4, fp32 accumulate): explicit 2D multicast configs over grids,
K-block widths and sub-blocks, against ttnn's default choice. Numerics are checked against the default output.

    python matmul_sweep.py --m 1024 --k 5184 --n 65536 [--transpose-b] [--top 12]
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import torch
import ttnn


def cfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )


def dev(a, device):
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(a)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def bench(fn, reps, device):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e3 / reps, out


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--k", type=int, default=5184)
    p.add_argument("--n", type=int, default=65536)
    p.add_argument("--transpose-b", action="store_true", help="B given as [N, K]")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--max-configs", type=int, default=400)
    args = p.parse_args()
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    M, K, N = args.m, args.k, args.n
    rng = np.random.default_rng(0)
    a = dev(rng.standard_normal((1, 1, M, K)).astype(np.float32), device)
    b_np = (rng.standard_normal((1, 1, N, K) if args.transpose_b else (1, 1, K, N)) * 0.02).astype(np.float32)
    b = dev(b_np, device)
    flops = 2 * M * K * N
    Mt, Kt, Nt = M // 32, K // 32, N // 32

    ms, ref = bench(
        lambda: ttnn.matmul(a, b, transpose_b=args.transpose_b, compute_kernel_config=cfg()), args.reps, device
    )
    ref_host = ttnn.to_torch(ref).float()
    print(
        f"M={M} K={K} N={N} transpose_b={args.transpose_b} grid {grid.x}x{grid.y}: default config {ms:.2f} ms, {flops / ms / 1e9:.1f} TFLOP/s wall"
    )
    ms_full, out = bench(
        lambda: ttnn.matmul(
            a, b, transpose_b=args.transpose_b, compute_kernel_config=cfg(), core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x)
        ),
        args.reps,
        device,
    )
    print(f"core_grid full: {ms_full:.2f} ms; 8x11: ", end="")
    ms_88, out = bench(
        lambda: ttnn.matmul(
            a, b, transpose_b=args.transpose_b, compute_kernel_config=cfg(), core_grid=ttnn.CoreGrid(y=8, x=11)
        ),
        args.reps,
        device,
    )
    print(f"{ms_88:.2f} ms")

    results = []
    grids = [(grid.x, grid.y), (11, 8), (8, 11), (10, 10), (8, 10), (10, 8), (8, 8), (11, 9), (9, 11)]
    tried = 0
    for gx, gy in grids:
        if gx > grid.x or gy > grid.y:
            continue
        for transpose_mcast in (False, True):
            # M split over gy (or gx when transposed), N over gx (or gy)
            m_cores, n_cores = (gx, gy) if transpose_mcast else (gy, gx)
            if Mt % m_cores and Nt % n_cores:
                pass
            per_core_m = -(-Mt // m_cores)
            per_core_n = -(-Nt // n_cores)
            for in0_block_w in [d for d in divisors(Kt) if d <= 16]:
                for sub_h, sub_w in ((1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (1, 8), (2, 4), (4, 2)):
                    if sub_h * sub_w > 8 or per_core_m % sub_h or per_core_n % sub_w:
                        continue
                    if tried >= args.max_configs:
                        break
                    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=in0_block_w,
                        out_subblock_h=sub_h,
                        out_subblock_w=sub_w,
                        per_core_M=per_core_m,
                        per_core_N=per_core_n,
                        transpose_mcast=transpose_mcast,
                        fused_activation=None,
                    )
                    tried += 1
                    try:
                        t, out = bench(
                            lambda: ttnn.matmul(
                                a, b, transpose_b=args.transpose_b, program_config=pc, compute_kernel_config=cfg()
                            ),
                            args.reps,
                            device,
                        )
                        err = float((ttnn.to_torch(out).float() - ref_host).abs().max())
                        ttnn.deallocate(out)
                        results.append(
                            (
                                t,
                                f"grid {gx}x{gy} tmcast={int(transpose_mcast)} in0_block_w={in0_block_w} per_core_M={per_core_m} N={per_core_n} sub={sub_h}x{sub_w}",
                                err,
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        results.append(
                            (
                                1e9,
                                f"grid {gx}x{gy} tmcast={int(transpose_mcast)} in0_block_w={in0_block_w} per_core_M={per_core_m} N={per_core_n} sub={sub_h}x{sub_w} FAILED {str(e).splitlines()[0][:60]}",
                                0.0,
                            )
                        )
    results.sort(key=lambda r: r[0])
    ok = [r for r in results if r[0] < 1e9]
    print(f"{tried} configs tried, {len(ok)} ran")
    for t, name, err in ok[: args.top]:
        print(f"  {t:8.2f} ms  {flops / t / 1e9:7.1f} TFLOP/s  max|diff vs default| {err:.2e}  {name}")
    fails = [r for r in results if r[0] >= 1e9]
    print(f"{len(fails)} failed; first: {fails[0][1] if fails else '-'}")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
