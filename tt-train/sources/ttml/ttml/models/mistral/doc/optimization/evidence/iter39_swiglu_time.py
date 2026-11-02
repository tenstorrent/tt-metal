# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 39: the MLP interior on the transposed layouts against the iteration-38 path, standalone at the TP
shard shape (T = 1024, I = 8192, K = 5120 + 64). Forward: gate_up matmul (the committed 1D config vs the
transposed-orientation 2D config) and swiglu_packed_fw (token-major in, h^T out via DEST transposes vs packed^T in,
elementwise). Backward: swiglu_packed_bw (transposed_grad vs transposed_io) and the gate_up dX matmul (transpose_b
vs the plain operand). Outputs compared."""

import time

import numpy as np
import torch
import ttnn
import ttml
from ttml.modules.frozen_linear import (
    _compute_config,
    dx_matmul_kwargs,
    dx_plain_matmul_kwargs,
    forward_matmul_kwargs,
    forward_transposed_matmul_kwargs,
)

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T, I, K = 1024, 8192, 5120 + 64
cfg = _compute_config()


def dev(a):
    return ttnn.from_torch(
        torch.from_numpy(a.astype(np.float32)).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def bench_pair(fa, fb, reps=10, rounds=6):
    """Both variants in alternating rounds (the chip's clock drifts between runs: the box throttles per chip, so
    only back-to-back numbers compare); returns the per-variant median round time in us and the last outputs."""
    ta, tb = [], []
    out_a = out_b = None
    for _ in range(rounds):
        for fn, ts in ((fa, ta), (fb, tb)):
            fn()
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(reps):
                out = fn()
            ttnn.synchronize_device(device)
            ts.append((time.perf_counter() - t0) * 1e6 / reps)
            if fn is fa:
                out_a = out
            else:
                out_b = out
    return float(np.median(ta)), float(np.median(tb)), out_a, out_b


def diff(a, b):
    return float((ttnn.to_torch(a).float() - ttnn.to_torch(b).float()).abs().max())


x_cat = dev(rng.standard_normal((1, 1, T, K)))
w_cat = dev(rng.standard_normal((1, 1, K, 2 * I)) * 0.02)
w_cat_t = ttnn.transpose(w_cat, -2, -1)  # [2I, K]
# --- forward
us_gu38, us_gu39, gu, gu_t = bench_pair(
    lambda: ttnn.matmul(x_cat, w_cat, compute_kernel_config=cfg, **forward_matmul_kwargs(K - 64, 2 * I, T, k=K)),
    lambda: ttnn.matmul(
        w_cat_t, x_cat, transpose_b=True, compute_kernel_config=cfg, **forward_transposed_matmul_kwargs(K, 2 * I, T)
    ),
)
print(
    f"gate_up forward: 1D [T, K] x [K, 2I] {us_gu38:.0f} us; transposed 2D [2I, K] x [T, K]^T {us_gu39:.0f} us; max|diff| {diff(ttnn.transpose(gu_t, -2, -1), gu):.3e}"
)
gu_t = ttnn.transpose(gu, -2, -1)  # the same data for both SwiGLU modes
us_sw38, us_sw39, h38, h39 = bench_pair(
    lambda: ttml.ops.metal.swiglu_packed_fw(gu, None, I + 64, transposed_output=True),
    lambda: ttml.ops.metal.swiglu_packed_fw(gu_t, None, I + 64, transposed_input=True),
)
print(
    f"swiglu_packed_fw -> h^T [I + 64, T]: from token-major gu (DEST transposes) {us_sw38:.0f} us; from gu^T (elementwise) {us_sw39:.0f} us; max|diff| {diff(h38, h39):.3e}"
)
print(f"forward interior: iteration 38 {us_gu38 + us_sw38:.0f} us; iteration 39 {us_gu39 + us_sw39:.0f} us")
# --- backward
dcat_t = dev(rng.standard_normal((1, 1, I + 64, T)) * 0.01)  # the w2 backward's product: [dh^T ; s_dh^T ; zeros]
a = dev(rng.standard_normal((1, 1, 16, I)) * 0.02)
scale = 1.5
us_bw38, us_bw39, dp38, dp39 = bench_pair(
    lambda: ttml.ops.metal.swiglu_packed_bw(gu, dcat_t, None, lora_a=a, transposed_grad=True, lowrank_scale=scale),
    lambda: ttml.ops.metal.swiglu_packed_bw(gu_t, dcat_t, None, lora_a=a, lowrank_scale=scale, transposed_io=True),
)
print(
    f"swiglu_packed_bw: transposed_grad (dpacked token-major) {us_bw38:.0f} us; transposed_io (dpacked^T) {us_bw39:.0f} us; max|diff| {diff(ttnn.transpose(dp39, -2, -1), dp38):.3e}"
)
dp39 = ttnn.transpose(dp38, -2, -1)  # the same data for both dX forms
us_dx38, us_dx39, dx38, dx39 = bench_pair(
    lambda: ttnn.matmul(w_cat, dp38, transpose_b=True, compute_kernel_config=cfg, **dx_matmul_kwargs(K, 2 * I, T)),
    lambda: ttnn.matmul(w_cat, dp39, compute_kernel_config=cfg, **dx_plain_matmul_kwargs(K, 2 * I, T)),
)
print(
    f"gate_up dX^T [K, 2I] x dpacked: transpose_b (ttnn's choice) {us_dx38:.0f} us; plain dpacked^T (2D depth 8) {us_dx39:.0f} us; max|diff| {diff(dx38, dx39):.3e}"
)
us_dx39b, us_dx39c, _, _ = bench_pair(
    lambda: ttnn.matmul(w_cat, dp39, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=10, x=11)),
    lambda: ttnn.matmul(w_cat, dp39, compute_kernel_config=cfg, **dx_plain_matmul_kwargs(K, 2 * I, T)),
)
print(f"gate_up dX^T plain dpacked^T: ttnn's choice {us_dx39b:.0f} us; 2D depth 8 {us_dx39c:.0f} us")
print(f"backward interior: iteration 38 {us_bw38 + us_dx38:.0f} us; iteration 39 {us_bw39 + us_dx39:.0f} us")
ttml.autograd.AutoContext.get_instance().close_device()
