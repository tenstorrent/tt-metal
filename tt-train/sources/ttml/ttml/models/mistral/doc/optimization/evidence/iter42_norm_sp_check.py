# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 42 (sequence parallel): the fused residual-norm forward and the norm backward on the chip's quarter of
the token rows ([256, 5120], the shard between the reduce-scatter and the all-gather) against the full rows
([1024, 5120]), alternating rounds in one process; outputs vs numpy fp64."""

import time

import numpy as np
import torch
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
C, eps = 5120, 1e-5


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


g_np = bf16(rng.normal(1, 0.1, (1, 1, 1, C)))
g = dev(g_np)
cases = {}
for T in (1024, 256):
    r_np, d_np = bf16(rng.normal(0, 1, (1, 1, T, C))), bf16(rng.normal(0, 0.5, (1, 1, T, C)))
    dy_np, dres_np = bf16(rng.normal(0, 0.01, (1, 1, T, C))), bf16(rng.normal(0, 0.01, (1, 1, T, C)))
    r, d, dy, dres = dev(r_np), dev(d_np), dev(dy_np), dev(dres_np)
    d_t = ttnn.transpose(d, -2, -1)
    _, _, rms = ttml.ops.metal.add_rmsnorm_fw_allcore(r, dev(np.zeros((1, 1, T, C))), g, eps)
    s_ref = r_np + d_np
    out_ref = s_ref / np.sqrt((s_ref**2).mean(-1, keepdims=True) + eps) * g_np
    cases[f"fwd rows {T:4d}, token-major delta"] = (
        lambda r=r, d=d: ttml.ops.metal.add_rmsnorm_fw_allcore(r, d, g, eps),
        lambda o: host(o[1]),
        out_ref,
    )
    cases[f"fwd rows {T:4d}, transposed delta"] = (
        lambda r=r, d_t=d_t: ttml.ops.metal.add_rmsnorm_fw_allcore(r, d_t, g, eps, True),
        lambda o: host(o[1]),
        out_ref,
    )
    rms_np = host(rms)[..., :1]
    inv = 1.0 / rms_np
    gdy = g_np * dy_np
    dot = (r_np * gdy).sum(-1, keepdims=True)
    dx_ref = inv * (gdy - r_np * dot * inv * inv / C) + dres_np
    cases[f"bwd rows {T:4d} + residual grad"] = (
        lambda r=r, rms=rms, dy=dy, dres=dres: ttml.ops.metal.rmsnorm_bw_allcore(r, g, rms, dy, False, dres),
        lambda o: host(o[0]),
        dx_ref,
    )
    cases[f"bwd rows {T:4d} + residual grad, transposed out"] = (
        lambda r=r, rms=rms, dy=dy, dres=dres: ttml.ops.metal.rmsnorm_bw_allcore(
            r, g, rms, dy, False, dres, transposed_grad=True
        ),
        lambda o: host(o[0]),
        dx_ref,
    )
times = {k: [] for k in cases}
outs = {}
names = list(cases)
for _ in range(3):
    for k in names + names[::-1]:
        tt, out = timed(cases[k][0])
        times[k].append(tt)
        outs[k] = out
for k in names:
    got = cases[k][1](outs[k])
    ref = cases[k][2]
    err = np.abs(got - ref).max() / np.abs(ref).max()
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    print(f"{k:46s} median {np.median(times[k]):6.0f} us  min {min(times[k]):6.0f}  rel {err:.1e}  PCC {pcc:.7f}")
ttml.autograd.AutoContext.get_instance().close_device()
