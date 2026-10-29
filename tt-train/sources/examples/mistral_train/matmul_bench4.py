# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Round 4: the backward routes with and without an explicit core_grid, under the device profiler so the numbers
are kernel cycle-time like Tracy's, for the gate_up and w2 shapes."""

from __future__ import annotations

import time

import numpy as np
import torch
import ttnn

M = 1024


def bench(fn, reps, device):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e3 / reps, out


def cfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )


def dev(a, device):
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(a)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def main():
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    full = ttnn.CoreGrid(y=grid.y, x=grid.x)
    rng = np.random.default_rng(0)
    for name, (K, N) in {
        "gate_up": (5120, 65536),
        "w2": (32768, 5120),
        "out": (4096, 5120),
        "lm_head": (5120, 131072),
    }.items():
        w_np = (rng.standard_normal((1, 1, N, K)) * 0.02).astype(np.float32)
        w, wt = dev(w_np, device), dev(w_np.transpose(0, 1, 3, 2), device)
        g = dev((rng.standard_normal((1, 1, M, N)) * 0.01).astype(np.float32), device)
        tests = {
            "dX today g@W (ttml: core_grid full)": lambda: ttnn.matmul(
                g, w, compute_kernel_config=cfg(), core_grid=full
            ),
            "g@W_t^T no core_grid (iter2 as run)": lambda: ttnn.matmul(
                g, wt, transpose_b=True, compute_kernel_config=cfg()
            ),
            "g@W_t^T core_grid full": lambda: ttnn.matmul(
                g, wt, transpose_b=True, compute_kernel_config=cfg(), core_grid=full
            ),
            "(W_t@g^T)^T no core_grid (iter2 as run)": lambda: ttnn.transpose(
                ttnn.matmul(wt, g, transpose_b=True, compute_kernel_config=cfg()), -2, -1
            ),
            "(W_t@g^T)^T core_grid full": lambda: ttnn.transpose(
                ttnn.matmul(wt, g, transpose_b=True, compute_kernel_config=cfg(), core_grid=full), -2, -1
            ),
            "(W_t@g^T)^T core_grid 8x11": lambda: ttnn.transpose(
                ttnn.matmul(wt, g, transpose_b=True, compute_kernel_config=cfg(), core_grid=ttnn.CoreGrid(y=8, x=11)),
                -2,
                -1,
            ),
        }
        print(name, K, N)
        for tname, fn in tests.items():
            try:
                ms, out = bench(fn, 5 if N < 100000 else 3, device)
                ttnn.deallocate(out)
                print(f"  {tname:45s} {ms:7.2f} ms wall")
            except Exception as e:  # noqa: BLE001
                print(f"  {tname:45s} FAILED {str(e).splitlines()[0][:80]}")
        for t in (w, wt, g):
            ttnn.deallocate(t)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
