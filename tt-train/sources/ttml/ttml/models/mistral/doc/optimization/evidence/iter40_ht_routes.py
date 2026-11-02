# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 40: the w2 LoRA h^T = A_wide @ x^T ([32, 8256] @ [8256, T], the LoRA tail rows of the transposed wide
buffer) on ttnn (the step's route: 33 cores, 85 us in the step) against the K-split skinny_matmul with the
transposed x and the transposed output written straight into the buffer's tail rows; alternating rounds, outputs
compared."""

import time

import numpy as np
import torch
import ttnn
import ttml
from ttml.modules.lora import lora_matmul_config

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T, I, R = 1024, 8192, 16
cfg = lora_matmul_config()
full = ttnn.CoreGrid(y=device.compute_with_storage_grid_size().y, x=device.compute_with_storage_grid_size().x)


def dev(a):
    return ttnn.from_torch(
        torch.from_numpy(a.astype(np.float32)).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def timed(fn, reps=30):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps, out


x_t_np = rng.standard_normal((1, 1, I + 64, T))
x_t_np[:, :, I:, :] = 0.0
x_t = dev(x_t_np)
a_np = rng.standard_normal((1, 1, R, I)) * 0.02
a = dev(a_np)
a_wide = dev(np.pad(a_np, [(0, 0), (0, 0), (0, 32 - R), (0, 64)]))
variants = {
    "ttnn a_wide [32, I+64] @ x^T (step)": lambda: ttnn.matmul(a_wide, x_t, compute_kernel_config=cfg, core_grid=full),
    "skinny_matmul x_transposed, transpose_out": lambda: ttml.ops.metal.skinny_matmul(
        x_t, a, True, None, 0, 0, True, True
    ),
    "  + tail rows written in place": lambda: ttml.ops.metal.skinny_matmul(x_t, a, True, x_t, I // 32, 1, True, True),
}
times = {k: [] for k in variants}
outs = {}
names = list(variants)
for _ in range(4):
    for k in names + names[::-1]:
        t, out = timed(variants[k])
        times[k].append(t)
        outs[k] = out
ref = torch.from_numpy(a_np[0, 0]).double() @ torch.from_numpy(x_t_np[0, 0, :I, :]).double()
for k in names:
    o = ttnn.to_torch(outs[k]).double()[0, 0, :R, :]
    print(
        f"{k:44s} median {np.median(times[k]):6.0f} us  min {min(times[k]):6.0f}  max|diff| vs fp64 {float((o - ref).abs().max()):.2e} (rel {float((o - ref).abs().max() / ref.abs().max()):.1e})"
    )
tail = ttnn.to_torch(x_t).double()[0, 0, I : I + 32, :]
print(
    f"tail rows of the buffer after the in-place write: max|diff| vs fp64 {float((tail[:R] - ref).abs().max()):.2e}, rows {R}..31 max|.| {float(tail[R:].abs().max()):.2e}"
)
ttml.autograd.AutoContext.get_instance().close_device()
