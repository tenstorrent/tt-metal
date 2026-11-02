# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 41: rmsnorm_bw_allcore (the single-phase rows kernel) at the step shape [1024, 5120] with the residual
gradient, token-major and with the transposed gradient output, timed with a fixed ttnn matmul as the clock reference
(two builds, two processes); dx vs numpy fp64 on bf16 inputs.

    python iter41_norm_bw_check.py before|after
"""

import sys
import time

import numpy as np
import torch
import ttnn
import ttml

label = sys.argv[1] if len(sys.argv) > 1 else "run"
ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T, C, eps = 1024, 5120, 1e-5


def dev(a):
    return ttnn.from_torch(
        torch.from_numpy(a.astype(np.float32)).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def host(t):
    return ttnn.to_torch(t).double().numpy()


def bf16(a):
    return torch.from_numpy(a.astype(np.float32)).bfloat16().float().double().numpy()


def timed(fn, reps=30):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps, out


x_np = bf16(rng.normal(0, 1, (1, 1, T, C)))
g_np = bf16(rng.normal(1, 0.1, (1, 1, 1, C)))
dy_np, dres_np = bf16(rng.normal(0, 0.01, (1, 1, T, C))), bf16(rng.normal(0, 0.01, (1, 1, T, C)))
x, g, dy, dres = dev(x_np), dev(g_np), dev(dy_np), dev(dres_np)
_, _, rms = ttml.ops.metal.add_rmsnorm_fw_allcore(x, dev(np.zeros((1, 1, T, C))), g, eps)
mm_a = dev(rng.standard_normal((1, 1, 1024, 5184)))
mm_b = dev(rng.standard_normal((1, 1, 5184, 8192)) * 0.02)
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
# reference: dx = inv * (gamma * dy - x * sum(x * gamma * dy) * inv^2 / C) + dres, with the op's bf16 rms
rms_np = host(rms)[..., :1]
inv = 1.0 / rms_np
gdy = g_np * dy_np
dot = (x_np * gdy).sum(-1, keepdims=True)
dx_ref = inv * (gdy - x_np * dot * inv * inv / C) + dres_np
variants = {
    "reference matmul [1024, 5184] x [5184, 8192]": lambda: ttnn.matmul(
        mm_a, mm_b, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=10, x=11)
    ),
    "rmsnorm_bw_allcore + residual grad": lambda: ttml.ops.metal.rmsnorm_bw_allcore(x, g, rms, dy, False, dres),
    "rmsnorm_bw_allcore + residual grad, transposed out": lambda: ttml.ops.metal.rmsnorm_bw_allcore(
        x, g, rms, dy, False, dres, transposed_grad=True
    ),
}
times = {k: [] for k in variants}
outs = {}
names = list(variants)
for _ in range(4):
    for k in names + names[::-1]:
        t, out = timed(variants[k])
        times[k].append(t)
        outs[k] = out
for k in names:
    line = f"{label:7s} {k:52s} median {np.median(times[k]):6.0f} us  min {min(times[k]):6.0f}"
    if k.startswith("rmsnorm"):
        dx = host(outs[k][0])
        err = np.abs(dx - dx_ref).max() / np.abs(dx_ref).max()
        pcc = np.corrcoef(dx.ravel(), dx_ref.ravel())[0, 1]
        line += f"  dx rel {err:.1e}  PCC {pcc:.7f}"
        if "transposed" in k:
            dxt = host(outs[k][2])
            line += f"  dx^T == dx^T: {np.array_equal(dxt[0, 0], dx[0, 0].T)}"
    print(line)
ttml.autograd.AutoContext.get_instance().close_device()
