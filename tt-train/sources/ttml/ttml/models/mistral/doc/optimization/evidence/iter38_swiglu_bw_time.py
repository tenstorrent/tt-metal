# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 38: swiglu_packed_bw on the transposed gradient with the fused low-rank term against the plain
backward plus transpose_split (the iteration-37 path), at the TP w2 shard shape; outputs compared."""

import time

import numpy as np
import torch
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T, I, R = 1024, 8192, 32


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


packed = dev(rng.standard_normal((1, 1, T, 2 * I)))
dcat_t = dev(rng.standard_normal((1, 1, I + 64, T)) * 0.01)  # [dh^T ; s_dh^T (32 rows) ; zeros]
a = dev(rng.standard_normal((1, 1, 16, I)) * 0.02)  # [16, I] (rank 16; the tile pads rows 16.. with zeros)
scale = 1.5
# iteration-37 path: transpose_split (dx token-major with the low-rank term) then the plain backward
us37, dx37 = bench(lambda: ttml.ops.metal.transpose_split(dcat_t, I, 16, scale, I + 64, a)[0])
us_bw37, out37 = bench(lambda: ttml.ops.metal.swiglu_packed_bw(packed, dx37))
# iteration-38 path: the transposed backward with the fused low-rank term
us38, out38 = bench(
    lambda: ttml.ops.metal.swiglu_packed_bw(packed, dcat_t, None, lora_a=a, transposed_grad=True, lowrank_scale=scale)
)
d = (ttnn.to_torch(out37).float() - ttnn.to_torch(out38).float()).abs()
print(f"iteration 37 path: transpose_split {us37:.0f} us + swiglu_packed_bw {us_bw37:.0f} us = {us37 + us_bw37:.0f} us")
print(f"iteration 38 path: swiglu_packed_bw(transposed_grad) {us38:.0f} us")
print(f"max|diff| between the two dpacked: {float(d.max()):.3e} (mean {float(d.mean()):.2e})")
ttml.autograd.AutoContext.get_instance().close_device()
