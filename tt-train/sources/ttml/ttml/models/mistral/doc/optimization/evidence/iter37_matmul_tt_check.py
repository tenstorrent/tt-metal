# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 37 scoping: the LoRA dB product with the output gradient in the transposed layout:
dB^T [r, out] = h^T @ g = ttnn.matmul(h [T, r], g_t [out, T], transpose_a=True, transpose_b=True) against
the current ttnn.matmul(h, g [T, out], transpose_a=True); correctness vs torch and time (HiFi4 / fp32 acc)."""

import time

import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
rng = np.random.default_rng(0)
T, R, OUT = 1024, 32, 5120
h_t = torch.from_numpy(rng.standard_normal((1, 1, T, R)).astype(np.float32)).bfloat16()
g_t = torch.from_numpy((rng.standard_normal((1, 1, T, OUT)) * 0.01).astype(np.float32)).bfloat16()
h = ttnn.from_torch(h_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
g = ttnn.from_torch(g_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
gT = ttnn.from_torch(g_t.transpose(-1, -2).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
ref = h_t.float().transpose(-1, -2) @ g_t.float()


def bench(fn, reps=20):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps, out


us0, o0 = bench(lambda: ttnn.matmul(h, g, transpose_a=True, compute_kernel_config=cfg))
us1, o1 = bench(lambda: ttnn.matmul(h, gT, transpose_a=True, transpose_b=True, compute_kernel_config=cfg))
for name, us, o in (("current  h^T @ g       ", us0, o0), ("proposed h^T @ (g^T)^T ", us1, o1)):
    d = (ttnn.to_torch(o).float() - ref).abs()
    print(
        f"{name}: {us:.0f} us, max|diff| vs torch {float(d.max()):.3e} (rel {float(d.max() / ref.abs().max()):.2e})",
        flush=True,
    )
print(f"both outputs identical: {bool((ttnn.to_torch(o0).float() == ttnn.to_torch(o1).float()).all())}")
ttnn.close_device(device)
