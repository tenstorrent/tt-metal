# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 39: the gate_up dB = s g^T h on the transposed gradient g^T [16384, T] (h [T, 32]): the all-core op's
transposed-big mode (what the step runs: 0.474 ms per 4 layers against the token-major ttnn route's 0.369) against
ttnn's plain [N, T] @ [T, 32] matmul (the [N, r] result needs no transpose) and ttnn's transpose_b route, in
alternating rounds; outputs compared."""

import time

import numpy as np
import torch
import ttnn
import ttml
from ttml.modules.lora import lora_matmul_config

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T, N, R = 1024, 16384, 16
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


g_t = dev(rng.standard_normal((1, 1, N, T)) * 0.01)
h = dev(np.pad(rng.standard_normal((1, 1, T, R)), [(0, 0), (0, 0), (0, 0), (0, 32 - R)]))
scale = 2.0
variants = {
    "op big_transposed (step)": lambda: ttml.ops.metal.skinny_contract(h, g_t, scale, True, big_transposed=True),
    "ttnn plain [N, T] @ [T, 32]": lambda: ttnn.multiply(
        ttnn.matmul(g_t, h, compute_kernel_config=cfg, core_grid=full), scale
    ),
    "ttnn plain, 2D 11x10": lambda: ttnn.multiply(
        ttnn.matmul(
            g_t,
            h,
            compute_kernel_config=cfg,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(11, 10),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=52,
                out_block_h=13,
                per_core_N=1,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
        ),
        scale,
    ),
    "skinny_matmul g^T [N, T] @ (h^T [32, T])^T (K-split op)": lambda: ttnn.multiply(
        ttml.ops.metal.skinny_matmul(g_t, ttnn.transpose(h, -2, -1), True, None, 0, 0), scale
    ),
    "ttnn transpose_b [32, T] x [N, T]^T + T": lambda: ttnn.transpose(
        ttnn.multiply(
            ttnn.matmul(ttnn.transpose(h, -2, -1), g_t, transpose_b=True, compute_kernel_config=cfg, core_grid=full),
            scale,
        ),
        -2,
        -1,
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
ref = ttnn.to_torch(outs[names[0]]).float()[..., :R]
for k in names:
    o = ttnn.to_torch(outs[k]).float()[..., :R]
    print(
        f"{k:42s} median {np.median(times[k]):6.0f} us  min {min(times[k]):6.0f}  max|diff| vs op {float((o - ref).abs().max()):.2e}"
    )
ttml.autograd.AutoContext.get_instance().close_device()
