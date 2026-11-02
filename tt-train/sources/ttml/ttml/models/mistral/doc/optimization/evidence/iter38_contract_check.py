# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 38: skinny_contract on a transposed big operand (dA = s_dh^T x with x^T [8256, T]; dB^T = h^T g with
g^T [5120, T]) against the token-major op and ttnn's transpose_b route; outputs compared (bit-exact expected)."""

import time

import numpy as np
import torch
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T, R, IN, OUT = 1024, 16, 8256, 5120
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


T_ = lambda t: ttnn.to_torch(t).float()
for name, N in (("dA (N = in + tail)", IN), ("dB (N = out)", OUT)):
    small = dev(rng.standard_normal((1, 1, T, R)) * 0.01)  # [T, r]
    big = dev(rng.standard_normal((1, 1, T, N)))  # [T, N]
    big_t = ttnn.transpose(big, -2, -1)  # [N, T]
    small_t = ttnn.transpose(small, -2, -1)  # [r, T]
    us0, o0 = bench(lambda: ttml.ops.metal.skinny_contract(small, big, 1.0, False))
    us1, o1 = bench(lambda: ttml.ops.metal.skinny_contract(small, big_t, 1.0, False, big_transposed=True))
    us2, o2 = bench(
        lambda: ttml.ops.metal.skinny_contract(small_t, big_t, 1.0, False, big_transposed=True, small_transposed=True)
    )
    us3, o3 = bench(lambda: ttnn.matmul(small_t, big_t, transpose_b=True, compute_kernel_config=cfg))
    d1 = float((T_(o0) - T_(o1)).abs().max())
    d2 = float((T_(o0) - T_(o2)).abs().max())
    d3 = float((T_(o0) - T_(o3)).abs().max())
    print(
        f"{name}: op token-major {us0:.0f} us | op big_transposed {us1:.0f} us (diff {d1:.1e}) | + small_transposed {us2:.0f} us (diff {d2:.1e}) | ttnn transpose_b {us3:.0f} us (diff {d3:.1e})",
        flush=True,
    )
ttml.autograd.AutoContext.get_instance().close_device()
