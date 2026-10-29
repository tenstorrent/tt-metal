# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Round 3: every base-linear shape of Mistral-Small-24B (single chip and TP=4 local shards), forward and dX,
today's layout (W [N,K], transpose_b) vs the weight stored transposed (W_t [K,N]) on the full grid and on 8x11,
and the dX^T route (W_t @ g^T, then a transpose) with the transpose cost included.

    python matmul_bench3.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
import ttnn

SHAPES = {
    "qkv": (5120, 6144),
    "out": (4096, 5120),
    "gate_up": (5120, 65536),
    "w2": (32768, 5120),
    "lm_head": (5120, 131072),
    "tp4 qkv": (5120, 1536),
    "tp4 out": (1024, 5120),
    "tp4 gate_up": (5120, 16384),
    "tp4 w2": (8192, 5120),
    "tp4 lm_head": (5120, 32768),
}
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
    full, g88 = ttnn.CoreGrid(y=grid.y, x=grid.x), ttnn.CoreGrid(y=8, x=11)
    rng = np.random.default_rng(0)
    print("shape | fwd today | fwd W_t full | fwd W_t 8x11 | dX today | dX^T route (incl. transpose) | ms")
    for name, (K, N) in SHAPES.items():
        x = dev(rng.standard_normal((1, 1, M, K)).astype(np.float32), device)
        w_np = (rng.standard_normal((1, 1, N, K)) * 0.02).astype(np.float32)
        w, wt = dev(w_np, device), dev(w_np.transpose(0, 1, 3, 2), device)
        g = dev((rng.standard_normal((1, 1, M, N)) * 0.01).astype(np.float32), device)
        reps = 6 if N * K < 4e8 else 4
        res = []
        for fn in (
            lambda: ttnn.linear(x, w, transpose_b=True, compute_kernel_config=cfg(), core_grid=full),
            lambda: ttnn.matmul(x, wt, compute_kernel_config=cfg(), core_grid=full),
            lambda: ttnn.matmul(x, wt, compute_kernel_config=cfg(), core_grid=g88),
            lambda: ttnn.matmul(g, w, compute_kernel_config=cfg(), core_grid=full),
            lambda: ttnn.transpose(
                ttnn.matmul(wt, g, transpose_b=True, compute_kernel_config=cfg(), core_grid=full), -2, -1
            ),
        ):
            try:
                ms, out = bench(fn, reps, device)
                ttnn.deallocate(out)
                res.append(f"{ms:7.2f}")
            except Exception as e:  # noqa: BLE001
                res.append("  FAIL ")
        print(f"{name:12s} K={K:6d} N={N:6d} | " + " | ".join(res))
        for t in (x, w, wt, g):
            ttnn.deallocate(t)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
