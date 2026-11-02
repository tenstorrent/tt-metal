# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 38: the w2 LoRA products on the transposed operands (h^T [32, T] in the wide^T buffer, s_dh^T [32, T],
g^T [out, T], x^T = the wide^T buffer [8256, T]): the orientation the first build used against the alternatives,
HiFi4 / fp32 accumulate; times and max |diff| vs torch fp32."""

import time

import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
rng = np.random.default_rng(0)
T, R, IN, OUT = 1024, 32, 8256, 5120
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)


def dev(a):
    return ttnn.from_torch(
        torch.from_numpy(a.astype(np.float32)).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def bench(fn, reps=20):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps, out


x_t = dev(rng.standard_normal((1, 1, IN, T)))  # wide^T: x^T (rows 0..8191 h^T, tail rows)
s_dh_t = dev(rng.standard_normal((1, 1, R, T)) * 0.01)  # s_dh^T
g_t = dev(rng.standard_normal((1, 1, OUT, T)) * 0.01)  # g^T
h_t = dev(rng.standard_normal((1, 1, R, T)))  # h^T (LoRA)
a_wide = dev(np.concatenate([rng.standard_normal((1, 1, R, IN - 64)) * 0.02, np.zeros((1, 1, R, 64))], axis=3))
full = ttnn.CoreGrid(y=grid.y, x=grid.x)
T_ = lambda t: ttnn.to_torch(t).float()
ref_da = T_(s_dh_t) @ T_(x_t).transpose(-1, -2)  # [32, 8256]
ref_db = T_(h_t) @ T_(g_t).transpose(-1, -2)  # [32, 5120]
ref_h = T_(a_wide) @ T_(x_t)  # [32, T]
cases = [
    (
        "dA  (i)  s_dh^T @ (x^T)^T, transpose_b (first build)",
        lambda: ttnn.matmul(s_dh_t, x_t, transpose_b=True, compute_kernel_config=cfg, core_grid=full),
        lambda o: (T_(o) - ref_da),
    ),
    (
        "dA  (ii) (x^T @ s_dh)^T: tall-skinny + transpose",
        lambda: ttnn.transpose(
            ttnn.matmul(x_t, ttnn.transpose(s_dh_t, -2, -1), compute_kernel_config=cfg, core_grid=full), -2, -1
        ),
        lambda o: (T_(o) - ref_da),
    ),
    (
        "dB  (i)  h^T @ (g^T)^T, transpose_b (first build)",
        lambda: ttnn.matmul(h_t, g_t, transpose_b=True, compute_kernel_config=cfg, core_grid=full),
        lambda o: (T_(o) - ref_db),
    ),
    (
        "dB  (ii) (g^T @ h)^T: tall-skinny + transpose",
        lambda: ttnn.transpose(
            ttnn.matmul(g_t, ttnn.transpose(h_t, -2, -1), compute_kernel_config=cfg, core_grid=full), -2, -1
        ),
        lambda o: (T_(o) - ref_db),
    ),
    (
        "h^T (i)  A_wide @ x^T, ttnn's choice (first build)",
        lambda: ttnn.matmul(a_wide, x_t, compute_kernel_config=cfg, core_grid=full),
        lambda o: (T_(o) - ref_h),
    ),
    (
        "h^T (ii) (x^T)^T @ A_wide^T = transpose_a / transpose_b",
        lambda: ttnn.transpose(
            ttnn.matmul(x_t, a_wide, transpose_a=True, transpose_b=True, compute_kernel_config=cfg, core_grid=full),
            -2,
            -1,
        ),
        lambda o: (T_(o) - ref_h),
    ),
]
for name, fn, err in cases:
    try:
        us, out = bench(fn)
        d = err(out).abs()
        print(f"{name:58s}: {us:6.0f} us  max|diff| {float(d.max()):.3e}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{name:58s}: FAILED {str(e).splitlines()[0][:100]}", flush=True)
ttnn.close_device(device)
