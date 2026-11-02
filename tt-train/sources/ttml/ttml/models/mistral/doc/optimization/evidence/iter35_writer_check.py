# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 35: numerics (against a torch fp32 product of the bf16-rounded operands) and timing of ttnn's own
matmul choice on the full grid for the model's base-matmul shapes, HiFi4 / fp32 accumulate / packer L1 accumulate.
Run before and after the writer-kernel change; the error must not move and the time must drop.

    python iter35_writer_check.py [--reps 10]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import ttnn

SHAPES = [  # (name, M, K, N, transpose_b)
    ("gate_up fwd", 1024, 5184, 16384, False),
    ("gate_up dX^T", 5184, 16384, 1024, True),
    ("w2 fwd", 1024, 8256, 5120, False),
    ("w2 dX^T", 8256, 5120, 1024, True),
    ("qkv fwd", 1024, 5184, 1792, False),
    ("out fwd", 1024, 1344, 5120, False),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )
    rng = np.random.default_rng(0)
    for name, M, K, N, tb in SHAPES:
        a_t = torch.from_numpy(rng.standard_normal((1, 1, M, K)).astype(np.float32)).bfloat16()
        b_t = torch.from_numpy(
            (rng.standard_normal((1, 1, N, K) if tb else (1, 1, K, N)) * 0.02).astype(np.float32)
        ).bfloat16()
        a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        def fn():
            return ttnn.matmul(
                a, b, transpose_b=tb, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x)
            )

        out = fn()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            out = fn()
        ttnn.synchronize_device(device)
        ms = (time.perf_counter() - t0) * 1e3 / args.reps
        ref = a_t.float() @ (b_t.float().transpose(-1, -2) if tb else b_t.float())
        got = ttnn.to_torch(out).float()
        err = (got - ref).abs()
        rel = float(err.max() / ref.abs().max())
        print(
            f"{name:>13} M={M} K={K} N={N}{' ^T' if tb else '   '}: {ms:.3f} ms {2 * M * K * N / ms / 1e9:6.1f} TFLOP/s"
            f"  max|diff| {float(err.max()):.3e} (rel to max {rel:.2e}), mean|diff| {float(err.mean()):.2e}",
            flush=True,
        )
        ttnn.deallocate(out)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
