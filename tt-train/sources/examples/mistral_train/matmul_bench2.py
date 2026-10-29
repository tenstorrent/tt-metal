# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Round 2: the backward dX shape with a pre-transposed weight, explicit 2D-multicast program configs for the
forward, and the LoRA B matmul with B stored transposed.

    python matmul_bench2.py --n 65536
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


def cfg(fidelity=ttnn.MathFidelity.HiFi4, fp32=True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32, math_approx_mode=False, packer_l1_acc=True
    )


def dev_tensor(a, device):
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(a)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=65536)
    p.add_argument("--k", type=int, default=5120)
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--reps", type=int, default=6)
    args = p.parse_args()
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    M, K, N = args.m, args.k, args.n
    rng = np.random.default_rng(0)
    x = dev_tensor(rng.standard_normal((1, 1, M, K)).astype(np.float32), device)
    w_np = (rng.standard_normal((1, 1, N, K)) * 0.02).astype(np.float32)
    w = dev_tensor(w_np, device)
    wt = dev_tensor(w_np.transpose(0, 1, 3, 2), device)
    g = dev_tensor((rng.standard_normal((1, 1, M, N)) * 0.01).astype(np.float32), device)
    full = ttnn.CoreGrid(y=grid.y, x=grid.x)
    g88 = ttnn.CoreGrid(y=8, x=11)
    tests = []
    # backward dX [M,N] x [N,K]: today W [N,K] no transpose; with W_t [K,N] it needs transpose_b
    tests += [
        (
            "dX today: matmul(g, W[N,K]), full grid",
            2 * M * N * K,
            lambda: ttnn.matmul(g, w, compute_kernel_config=cfg(), core_grid=full),
        ),
        (
            "dX today: matmul(g, W[N,K]), 8x11",
            2 * M * N * K,
            lambda: ttnn.matmul(g, w, compute_kernel_config=cfg(), core_grid=g88),
        ),
        (
            "dX with W_t: matmul(g, W_t, transpose_b), full grid",
            2 * M * N * K,
            lambda: ttnn.matmul(g, wt, transpose_b=True, compute_kernel_config=cfg(), core_grid=full),
        ),
        (
            "dX with W_t: matmul(g, W_t, transpose_b), 8x11",
            2 * M * N * K,
            lambda: ttnn.matmul(g, wt, transpose_b=True, compute_kernel_config=cfg(), core_grid=g88),
        ),
        (
            "dX^T with W_t: matmul(W_t, g, transpose_b) -> [K,M]",
            2 * M * N * K,
            lambda: ttnn.matmul(wt, g, transpose_b=True, compute_kernel_config=cfg(), core_grid=full),
        ),
    ]
    # forward with explicit 2D multicast program configs on the full grid (M tiles 32, N tiles N/32)
    Mt, Nt, Kt = M // 32, N // 32, K // 32
    for per_core_m, in0_block_w, sub_n in ((4, 4, 2), (4, 8, 2), (8, 4, 2), (4, 4, 4), (2, 4, 2)):
        for gy, gx in ((grid.y, grid.x), (8, 11)):
            per_core_n = -(-Nt // gx)
            if Mt % per_core_m or per_core_n % sub_n:
                continue
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=sub_n,
                per_core_M=per_core_m,
                per_core_N=per_core_n,
                transpose_mcast=False,
                fused_activation=None,
            )
            tests.append(
                (
                    f"fwd W_t, 2D mcast grid {gx}x{gy} per_core_M={per_core_m} N={per_core_n} in0_block_w={in0_block_w} sub_n={sub_n}",
                    2 * M * K * N,
                    (lambda pc=pc: ttnn.matmul(x, wt, program_config=pc, compute_kernel_config=cfg())),
                )
            )
    # LoRA B matmul: h [M,32] x B^T, B [N,32] (today) vs B_t [32,N]
    h = dev_tensor(rng.standard_normal((1, 1, M, 32)).astype(np.float32), device)
    b = dev_tensor((rng.standard_normal((1, 1, N, 32)) * 0.02).astype(np.float32), device)
    bt = dev_tensor((rng.standard_normal((1, 1, 32, N)) * 0.02).astype(np.float32), device)
    tests += [
        (
            "LoRA B today: matmul(h, B[N,32], transpose_b)",
            2 * M * 32 * N,
            lambda: ttnn.matmul(h, b, transpose_b=True, compute_kernel_config=cfg()),
        ),
        ("LoRA B_t: matmul(h, B_t[32,N])", 2 * M * 32 * N, lambda: ttnn.matmul(h, bt, compute_kernel_config=cfg())),
        ("LoRA dh today: matmul(g, B[N,32])", 2 * M * 32 * N, lambda: ttnn.matmul(g, b, compute_kernel_config=cfg())),
        (
            "LoRA dh with B_t: matmul(g, B_t, transpose_b)",
            2 * M * 32 * N,
            lambda: ttnn.matmul(g, bt, transpose_b=True, compute_kernel_config=cfg()),
        ),
        (
            "LoRA dB today: matmul(g^T, h) -> [N,32]",
            2 * M * 32 * N,
            lambda: ttnn.matmul(g, h, transpose_a=True, compute_kernel_config=cfg()),
        ),
        (
            "LoRA dB_t: matmul(h^T, g) -> [32,N]",
            2 * M * 32 * N,
            lambda: ttnn.matmul(h, g, transpose_a=True, compute_kernel_config=cfg()),
        ),
    ]
    print(f"M={M} K={K} N={N}, grid {grid.x}x{grid.y}")
    for name, flops, fn in tests:
        try:
            ms, out = bench(fn, args.reps, device)
            print(f"  {name:70s} {ms:8.2f} ms  {flops / ms / 1e9:7.1f} TFLOP/s wall")
            ttnn.deallocate(out)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:70s} FAILED: {str(e).splitlines()[0][:100]}")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
