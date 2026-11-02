# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 35: time ttnn's own matmul choice on the full core grid (what the model's dX / qkv / out paths call),
HiFi4 / fp32 accumulate / packer L1 accumulate, for the kernel attribution variants.

    python iter35_mm_time.py --m 5184 --k 16384 --n 1024 --transpose-b [--reps 10]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import ttnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--transpose-b", action="store_true")
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    M, K, N = args.m, args.k, args.n
    rng = np.random.default_rng(0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )

    def dev(a):
        return ttnn.from_torch(torch.from_numpy(a), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    a = dev(rng.standard_normal((1, 1, M, K)).astype(np.float32))
    b = dev((rng.standard_normal((1, 1, N, K) if args.transpose_b else (1, 1, K, N)) * 0.02).astype(np.float32))

    def fn():
        return ttnn.matmul(
            a, b, transpose_b=args.transpose_b, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x)
        )

    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(args.reps):
        out = fn()
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) * 1e3 / args.reps
    flops = 2 * M * K * N
    print(
        f"M={M} K={K} N={N}{' transpose_b' if args.transpose_b else ''}: ttnn's choice on {grid.x}x{grid.y}: "
        f"{ms:.3f} ms, {flops / ms / 1e9:.1f} TFLOP/s wall"
    )
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
