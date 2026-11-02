# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 39: the SwiGLU forward/backward modes in ABBA order (order bias check), same data, TP shard shape."""

import time

import numpy as np
import torch
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T, I = 1024, 8192


def dev(a):
    return ttnn.from_torch(
        torch.from_numpy(a.astype(np.float32)).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def timed(fn, reps=30):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps


gu = dev(rng.standard_normal((1, 1, T, 2 * I)))
gu_t = ttnn.transpose(gu, -2, -1)
dcat_t = dev(rng.standard_normal((1, 1, I + 64, T)) * 0.01)
a = dev(rng.standard_normal((1, 1, 16, I)) * 0.02)
variants = {
    "fw38 token-major in, h^T out": lambda: ttml.ops.metal.swiglu_packed_fw(gu, None, I + 64, transposed_output=True),
    "fw39 gu^T in, h^T out": lambda: ttml.ops.metal.swiglu_packed_fw(gu_t, None, I + 64, transposed_input=True),
    "bw38 transposed_grad": lambda: ttml.ops.metal.swiglu_packed_bw(
        gu, dcat_t, None, lora_a=a, transposed_grad=True, lowrank_scale=1.5
    ),
    "bw39 transposed_io": lambda: ttml.ops.metal.swiglu_packed_bw(
        gu_t, dcat_t, None, lora_a=a, lowrank_scale=1.5, transposed_io=True
    ),
}
times = {k: [] for k in variants}
names = list(variants)
for _ in range(4):
    for k in names + names[::-1]:  # ABCD DCBA
        times[k].append(timed(variants[k]))
for k in names:
    print(f"{k:34s} median {np.median(times[k]):6.0f} us  min {min(times[k]):6.0f}  max {max(times[k]):6.0f}")
ttml.autograd.AutoContext.get_instance().close_device()
