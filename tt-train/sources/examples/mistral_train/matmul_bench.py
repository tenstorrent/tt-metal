# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark of the Mistral gate_up / LM-head forward matmul shape on one chip.

``[1024, 5120] x W^T`` with ``W = [N, 5120]`` (N = 65536 or 131072) as ttml's linear issues it (transpose_b,
HiFi4, fp32 accumulate, full core grid) against variants: pre-transposed weight (no transpose_b), ttnn.linear vs
ttnn.matmul, core grid 8x11 vs 10x11, explicit 2D multicast program configs, fp32 accumulate off, HiFi2.
Wall time over repetitions with a device sync (cycle-time scales 1:1 with the kernel at this size).

    python matmul_bench.py --n 65536 --reps 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import ttnn


def bench(fn, reps, device):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) * 1e3 / reps
    return ms, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=65536)
    p.add_argument("--k", type=int, default=5120)
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    M, K, N = args.m, args.k, args.n
    flops = 2 * M * K * N
    rng = np.random.default_rng(0)
    x = ttnn.from_torch if False else None  # noqa
    x_np = rng.standard_normal((1, 1, M, K)).astype(np.float32)
    w_np = (rng.standard_normal((1, 1, N, K)) * 0.02).astype(np.float32)
    x = (
        ttnn.from_numpy(x_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if hasattr(ttnn, "from_numpy")
        else None
    )
    if x is None:
        import torch

        x = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        w = ttnn.from_torch(torch.from_numpy(w_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        wt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(w_np.transpose(0, 1, 3, 2))),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    else:
        w = ttnn.from_numpy(w_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        wt = ttnn.from_numpy(
            np.ascontiguousarray(w_np.transpose(0, 1, 3, 2)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def cfg(fidelity=ttnn.MathFidelity.HiFi4, fp32=True):
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity, fp32_dest_acc_en=fp32, math_approx_mode=False, packer_l1_acc=True
        )

    full = ttnn.CoreGrid(y=grid.y, x=grid.x)
    g88 = ttnn.CoreGrid(y=8, x=11)
    variants = {
        "linear transpose_b, full grid (ttml today)": lambda: ttnn.linear(
            x, w, transpose_b=True, compute_kernel_config=cfg(), core_grid=full
        ),
        "linear transpose_b, 8x11 grid": lambda: ttnn.linear(
            x, w, transpose_b=True, compute_kernel_config=cfg(), core_grid=g88
        ),
        "matmul pre-transposed W [K,N], full grid": lambda: ttnn.matmul(
            x, wt, compute_kernel_config=cfg(), core_grid=full
        ),
        "matmul pre-transposed W, 8x11 grid": lambda: ttnn.matmul(x, wt, compute_kernel_config=cfg(), core_grid=g88),
        "matmul pre-transposed W, no core_grid (ttnn default)": lambda: ttnn.matmul(x, wt, compute_kernel_config=cfg()),
        "linear transpose_b, no core_grid": lambda: ttnn.linear(x, w, transpose_b=True, compute_kernel_config=cfg()),
        "matmul pre-transposed W, full grid, fp32 acc off": lambda: ttnn.matmul(
            x, wt, compute_kernel_config=cfg(fp32=False), core_grid=full
        ),
        "matmul pre-transposed W, full grid, HiFi2": lambda: ttnn.matmul(
            x, wt, compute_kernel_config=cfg(ttnn.MathFidelity.HiFi2), core_grid=full
        ),
    }
    ref = None
    print(f"M={M} K={K} N={N}, grid {grid.x}x{grid.y}, {flops / 1e9:.1f} GFLOP per call, {args.reps} reps")
    for name, fn in variants.items():
        try:
            ms, out = bench(fn, args.reps, device)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:55s} FAILED: {str(e).splitlines()[0][:120]}")
            continue
        host = ttnn.to_torch(out).float() if ref is None or True else None
        if ref is None:
            ref = host
            err = 0.0
        else:
            err = float((host - ref).norm() / ref.norm())
        print(f"  {name:55s} {ms:8.2f} ms  {flops / ms / 1e9:7.1f} TFLOP/s wall  rel err vs first {err:.1e}")
        ttnn.deallocate(out)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
