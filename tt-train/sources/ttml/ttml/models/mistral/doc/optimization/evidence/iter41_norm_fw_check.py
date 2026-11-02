# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 41: add_rmsnorm_fw_allcore (the single-phase rows kernel) at the step shape [1024, 5120] with the
token-major and the transposed delta, timed with a fixed ttnn matmul as the clock reference (the two builds run in
separate processes: the reference tells the clock drift apart from the change); outputs vs numpy fp64 on bf16 inputs.

    python iter41_norm_fw_check.py before|after
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


r_np, d_np = bf16(rng.normal(0, 1, (1, 1, T, C))), bf16(rng.normal(0, 0.5, (1, 1, T, C)))
g_np = bf16(rng.normal(1, 0.1, (1, 1, 1, C)))
r, d, g = dev(r_np), dev(d_np), dev(g_np)
d_t = ttnn.transpose(d, -2, -1)
mm_a = dev(rng.standard_normal((1, 1, 1024, 5184)))
mm_b = dev(rng.standard_normal((1, 1, 5184, 8192)) * 0.02)
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
s_ref = r_np + d_np
out_ref = s_ref / np.sqrt((s_ref**2).mean(-1, keepdims=True) + eps) * g_np
variants = {
    "reference matmul [1024, 5184] x [5184, 8192]": lambda: ttnn.matmul(
        mm_a, mm_b, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=10, x=11)
    ),
    "add_rmsnorm_fw_allcore, token-major delta": lambda: ttml.ops.metal.add_rmsnorm_fw_allcore(r, d, g, eps),
    "add_rmsnorm_fw_allcore, transposed delta": lambda: ttml.ops.metal.add_rmsnorm_fw_allcore(r, d_t, g, eps, True),
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
    line = f"{label:7s} {k:46s} median {np.median(times[k]):6.0f} us  min {min(times[k]):6.0f}"
    if k.startswith("add_rmsnorm"):
        s_got, out_got = host(outs[k][0]), host(outs[k][1])
        err_s = np.abs(s_got - s_ref).max() / np.abs(s_ref).max()
        err_o = np.abs(out_got - out_ref).max() / np.abs(out_ref).max()
        pcc = np.corrcoef(out_got.ravel(), out_ref.ravel())[0, 1]
        line += f"  sum rel {err_s:.1e}  out rel {err_o:.1e}  out PCC {pcc:.7f}"
    print(line)
ttml.autograd.AutoContext.get_instance().close_device()
