"""Iteration 25: transpose_split with the fused low-rank update vs transpose_split + lowrank_update, at the two
fused shapes (gate_up: in 5120; w2 shard: in 8192, wide dx), plus timings."""
import time

import numpy as np
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
metal = ttml.ops.metal


def dev_t(a):
    return ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16).get_value()


def host(t):
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32)


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


ok = True
for name, I, dx_cols, scale in (
    ("gate_up", 5120, 0, 1.0),
    ("w2 shard, wide dx", 8192, 8192 + 64, 1.0),
    ("w2 shard, scaled tail", 8192, 8192 + 64, 4.0),
):
    T, R, TAIL = 1024, 16, 64
    dcat = rng.normal(0, 1, (1, 1, I + TAIL, T))
    dcat[:, :, I + R :, :] = 0.0  # the zero tail rows of W_cat's gradient product
    dcat_t = dev_t(dcat)
    A = dev_t(rng.normal(0, 0.05, (1, 1, R, I)))
    dx_r, tail_r = metal.transpose_split(dcat_t, I, R, scale, dx_cols)
    up_r = metal.lowrank_update(dx_r, tail_r, A)
    dx_f, tail_f = metal.transpose_split(dcat_t, I, R, scale, dx_cols, A)
    a, b = host(up_r), host(dx_f)
    # torch-free fp64 reference for the dx part
    ref = np.swapaxes(dcat[..., :I, :], -1, -2) + (scale * np.swapaxes(dcat[..., I : I + R, :], -1, -2)).astype(
        np.float64
    ) @ host(A).astype(np.float64)
    d_old = np.abs(a[..., :I] - ref).max()
    d_new = np.abs(b[..., :I] - ref).max()
    d = np.abs(a[..., :I] - b[..., :I])
    print(
        f"{name}: fused vs two ops max|diff| {d.max():.3e} differing {(d > 0).sum()}/{d.size}; vs fp64 ref: two ops {d_old:.3e}, fused {d_new:.3e}",
        flush=True,
    )
    dt = np.abs(host(tail_f) - host(tail_r)).max()
    print(f"    tail: max|diff| {dt:.3e}", flush=True)
    ok &= dt == 0.0 and d_new <= d_old * 1.01 + 1e-6
    if dx_cols:
        dw = np.abs(b[..., I:] - a[..., I:]).max()
        print(f"    wide tail columns: max|diff| {dw:.3e}", flush=True)
    t_two = bench(
        lambda: metal.lowrank_update(*[metal.transpose_split(dcat_t, I, R, scale, dx_cols)[i] for i in (0, 1)], A)
    )
    t_ts = bench(lambda: metal.transpose_split(dcat_t, I, R, scale, dx_cols))
    t_fused = bench(lambda: metal.transpose_split(dcat_t, I, R, scale, dx_cols, A))
    print(f"    us: transpose_split {t_ts:.0f} + lowrank_update = {t_two:.0f} -> fused {t_fused:.0f}", flush=True)
print("FUSED LOWRANK", "PASS" if ok else "FAIL", flush=True)
ttml.autograd.AutoContext.get_instance().close_device()
