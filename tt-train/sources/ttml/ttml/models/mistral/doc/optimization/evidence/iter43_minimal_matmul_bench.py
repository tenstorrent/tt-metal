# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 43 scoping: ttnn's experimental minimal_matmul (a block-streaming matmul with its own config) against
the step's committed routes at the four big TP matmul shapes, one chip, alternating rounds; outputs compared to
the committed route. Shapes as the step runs them: gate_up forward in the transposed orientation (in1 given as
[K, T], the free in1 transpose of the mcast route is not available), gate_up dX^T, w2 forward (transposed
orientation), w2 dX^T.
"""

import time

import numpy as np
import torch
import ttnn
import ttml
from ttml.modules.frozen_linear import (
    _compute_config,
    dx_matmul_kwargs,
    dx_plain_matmul_kwargs,
    forward_transposed_matmul_kwargs,
)

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
cfg = _compute_config()
T = 1024


def dev(a):
    return ttnn.from_torch(
        torch.from_numpy(a.astype(np.float32)).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def timed(fn, reps=10):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps, out


def bench_pair(fa, fb, rounds=4):
    ta, tb = [], []
    oa = ob = None
    for _ in range(rounds):
        t, oa = timed(fa)
        ta.append(t)
        t, ob = timed(fb)
        tb.append(t)
    return float(np.median(ta)), float(np.median(tb)), oa, ob


def mmc(mb, kb, nb, sh, sw, gx, gy):
    return ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=kb,
        N_block_size=nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )


import itertools

sweep = [
    (mb, kb, nb, sh, sw, 11, 10)
    for mb, kb, nb, (sh, sw) in itertools.product((4, 8, 13, 16), (3, 6, 9, 18), (1, 3), ((4, 1), (2, 2), (1, 3)))
    if mb % sh == 0 and nb % sw == 0 and sh * sw <= 4
]
shapes = {
    "gate_up fwd^T [16384, 5184] x [5184, 1024]": (16384, 5184, 1024, "fwd_t", sweep),
    "w2 fwd^T [5120, 8256] x [8256, 1024]": (5120, 8256, 1024, "fwd_t", [c for c in sweep if c[1] != 9]),
}
for name, (M, K, N, route, configs) in shapes.items():
    a = dev(rng.standard_normal((1, 1, M, K)) * 0.02)
    b_kn = dev(rng.standard_normal((1, 1, K, N)))
    b_nk = ttnn.transpose(b_kn, -2, -1)
    if route == "fwd_t":
        kw = forward_transposed_matmul_kwargs(K, M, N) or {"core_grid": ttnn.CoreGrid(y=10, x=11)}
        committed = lambda: ttnn.matmul(a, b_nk, transpose_b=True, compute_kernel_config=cfg, **kw)
    elif route == "dx_plain":
        committed = lambda: ttnn.matmul(a, b_kn, compute_kernel_config=cfg, **dx_plain_matmul_kwargs(M, K, N))
    else:
        committed = lambda: ttnn.matmul(
            a, b_nk, transpose_b=True, compute_kernel_config=cfg, **dx_matmul_kwargs(M, K, N)
        )
    t_c, _ = timed(committed)
    print(f"{name}: committed route {t_c:.0f} us (first look)", flush=True)
    for c in configs:
        mc = mmc(*c)
        try:
            tc, tm, oc, om = bench_pair(
                committed, lambda: ttnn.experimental.minimal_matmul(a, b_kn, compute_kernel_config=cfg, config=mc)
            )
            d = float((ttnn.to_torch(om).float() - ttnn.to_torch(oc).float()).abs().max())
            print(
                f"  minimal_matmul M{c[0]} K{c[1]} N{c[2]} sub {c[3]}x{c[4]} grid {c[5]}x{c[6]}: {tm:.0f} us vs committed {tc:.0f} ({(tm / tc - 1) * 100:+.1f} %)  max|diff| {d:.2e}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            msg = [l for l in str(e).splitlines() if "FATAL" in l or "Error" in l]
            print(
                f"  minimal_matmul M{c[0]} K{c[1]} N{c[2]} sub {c[3]}x{c[4]} grid {c[5]}x{c[6]}: FAILED {(msg[0] if msg else str(e))[:140]}",
                flush=True,
            )
    ttnn.deallocate(a)
    ttnn.deallocate(b_kn)
    ttnn.deallocate(b_nk)
ttml.autograd.AutoContext.get_instance().close_device()
