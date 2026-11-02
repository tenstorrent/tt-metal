# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 40: the four rank-16 products of the transposed MLP interior, the step's routes before (iteration 39:
ttnn for the w2 trio, the pre-fix op for the gate_up dB) against the boosted own ops (the small operand multicast,
contiguous column ranges per core, the K-split skinny_matmul on the transposed buffer), alternating rounds; plus the
token-major routing table shapes (qkv / out dB) for the op against ttnn. Outputs compared to fp64."""

import time

import numpy as np
import torch
import ttnn
import ttml
from ttml.modules.lora import lora_matmul_config

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T, I, K, R, OUT = 1024, 8192, 5120, 16, 16384
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


def f64(a):
    return torch.from_numpy(a).double()


x_t_np = rng.standard_normal((1, 1, I + 64, T))
x_t_np[:, :, I:, :] = 0.0
a_np = rng.standard_normal((1, 1, R, I)) * 0.02
s_dh_np = np.zeros((1, 1, 32, T))
s_dh_np[:, :, :R, :] = rng.standard_normal((1, 1, R, T))
h_np = rng.standard_normal((1, 1, T, R))
h_t_np = np.ascontiguousarray(np.swapaxes(h_np, -1, -2))
g5k_np = rng.standard_normal((1, 1, K, T)) * 0.01
g16k_np = rng.standard_normal((1, 1, OUT, T)) * 0.01
x_t, a, s_dh, h, h_t, g5k, g16k = (dev(v) for v in (x_t_np, a_np, s_dh_np, h_np, h_t_np, g5k_np, g16k_np))
a_wide = dev(np.pad(a_np, [(0, 0), (0, 0), (0, 32 - R), (0, 64)]))
h_pad = dev(np.pad(h_np, [(0, 0), (0, 0), (0, 0), (0, 32 - R)]))
h_t_pad = dev(np.pad(h_t_np, [(0, 0), (0, 0), (0, 32 - R), (0, 0)]))  # the step's h^T [32, T] (iteration 38-39)
scale = 2.0
refs = {
    "h^T": f64(a_np[0, 0]) @ f64(x_t_np[0, 0, :I, :]),
    "dA": f64(s_dh_np[0, 0, :R, :]) @ f64(x_t_np[0, 0, :I, :]).T,
    "dB w2": scale * (f64(g5k_np[0, 0]) @ f64(h_np[0, 0])),
    "dB gate_up": scale * (f64(g16k_np[0, 0]) @ f64(h_np[0, 0])),
}


def before_ht():
    out = ttnn.matmul(a_wide, x_t, compute_kernel_config=cfg, core_grid=full)
    ttml.ops.metal.write_tile_block(x_t, out, I // 32, 0)
    return out


def before_db_w2():
    db_t = ttnn.multiply(ttnn.matmul(h_t_pad, g5k, transpose_b=True, compute_kernel_config=cfg, core_grid=full), scale)
    return ttnn.transpose(db_t, -2, -1)


pairs = {
    "h^T  [32, I+64] @ x^T": (
        ("ttnn + write_tile_block (39)", before_ht, lambda o: ttnn.to_torch(o).double()[0, 0, :R, :]),
        (
            "skinny_matmul x^T, tail rows in place",
            lambda: ttml.ops.metal.skinny_matmul(x_t, a, True, x_t, I // 32, 1, True, True),
            lambda o: ttnn.to_torch(o).double()[0, 0, :R, :],
        ),
    ),
    "dA   s_dh^T [32, T] x x^T^T": (
        (
            "ttnn transpose_b (39)",
            lambda: ttnn.matmul(s_dh, x_t, transpose_b=True, compute_kernel_config=cfg, core_grid=full),
            lambda o: ttnn.to_torch(o).double()[0, 0, :R, :I],
        ),
        (
            "op big^T small^T",
            lambda: ttml.ops.metal.skinny_contract(s_dh, x_t, 1.0, False, big_transposed=True, small_transposed=True),
            lambda o: ttnn.to_torch(o).double()[0, 0, :R, :I],
        ),
    ),
    "dB w2  h^T x g^T [5120, T]": (
        (
            "ttnn transpose_b + multiply + transpose (39)",
            before_db_w2,
            lambda o: ttnn.to_torch(o).double()[0, 0, :, :R],
        ),
        (
            "op big^T small^T tr_out",
            lambda: ttml.ops.metal.skinny_contract(h_t, g5k, scale, True, big_transposed=True, small_transposed=True),
            lambda o: ttnn.to_torch(o).double()[0, 0, :, :R],
        ),
    ),
    "dB gate_up  h x g^T [16384, T]": (
        ("op big^T (39: items of 4 columns, every core reads the small operand)", None, None),
        (
            "op big^T (40: multicast, column ranges)",
            lambda: ttml.ops.metal.skinny_contract(h, g16k, scale, True, big_transposed=True),
            lambda o: ttnn.to_torch(o).double()[0, 0, :, :R],
        ),
    ),
}
print(
    "product                              route                                              median us   max|diff| vs fp64 (rel)"
)
for name, (b, aft) in pairs.items():
    ref = refs[name.split()[0] if not name.startswith("dB") else ("dB w2" if "w2" in name else "dB gate_up")]
    for label, fn, view in (b, aft):
        if fn is None:
            print(f"{name:36s} {label:50s} 123 (O/iter40_contract_attrib.log)")
            continue
        times = []
        for _ in range(4):
            tt, out = timed(fn)
            times.append(tt)
        o = view(out)
        err = float((o - ref).abs().max())
        rel = err / float(ref.abs().max())
        print(f"{name:36s} {label:50s} {np.median(times):6.0f}      {err:.2e} ({rel:.1e})")
# the token-major routing table shapes (lora.py skinny_contract): op vs ttnn's transposed-a matmul + transpose
g_out = dev(rng.standard_normal((1, 1, T, K)) * 0.01)
g_qkv = dev(rng.standard_normal((1, 1, T, 1536)) * 0.01)
for label, g in (("out dB  N=5120", g_out), ("qkv dB  N=1536", g_qkv)):
    tb = [
        timed(
            lambda: ttnn.transpose(
                ttnn.matmul(h_pad, g, transpose_a=True, compute_kernel_config=cfg, core_grid=full), -2, -1
            )
        )[0]
        for _ in range(4)
    ]
    to = [timed(lambda: ttml.ops.metal.skinny_contract(h_t, g, 1.0, True, small_transposed=True))[0] for _ in range(4)]
    print(
        f"token-major {label}: ttnn transposed-a + transpose {np.median(tb):.0f} us; op (multicast) {np.median(to):.0f} us"
    )
ttml.autograd.AutoContext.get_instance().close_device()
