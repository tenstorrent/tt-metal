"""Correctness and wall time of ttml.ops.metal.transpose_split against ttnn slice + transpose (+ multiply)."""
import time

import numpy as np
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)


def dev_t(a):
    return ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16).get_value()


def host(t):
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)


def bench(fn, iters=30):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


def reference(x, rows, rank, scale):
    T = int(x.shape[-1])
    dx = ttnn.transpose(ttnn.slice(x, [0, 0, 0, 0], [1, 1, rows, T]), -2, -1)
    dh = ttnn.transpose(ttnn.slice(x, [0, 0, rows, 0], [1, 1, rows + rank, T]), -2, -1)
    if scale != 1.0:
        dh = ttnn.multiply(dh, scale)
    return dx, dh


# (in_features + TAIL rows, T, in_features, rank, dh_scale): gate_up (5120 + 64) and w2 (8192 + 64) at T=1024, TP 1/4
for R, T, rows, rank, s in [
    (5184, 1024, 5120, 16, 1.0),
    (8256, 1024, 8192, 16, 1.0),
    (5184, 1024, 5120, 16, 0.25),
    (544, 2048, 512, 32, 1.0),
]:
    x = dev_t(rng.normal(0, 1, (1, 1, R, T)))
    rdx, rdh = (host(t) for t in reference(x, rows, rank, s))
    gdx, gdh = (host(t) for t in ttml.ops.metal.transpose_split(x, rows, rank, s))
    for name, got, ref in (("dx", gdx, rdx), ("dh", gdh, rdh)):
        diff = np.abs(got - ref)
        print(
            f"R={R} T={T} rows={rows} rank={rank} scale={s} {name}: shape {got.shape} max|diff| {diff.max():.2e} differing {int((diff > 0).sum())}/{got.size}",
            flush=True,
        )
    t_ref = bench(lambda: reference(x, rows, rank, s))
    t_new = bench(lambda: ttml.ops.metal.transpose_split(x, rows, rank, s))
    print(f"  wall us ttnn slice+transpose {t_ref:.0f} transpose_split {t_new:.0f}", flush=True)
ttml.autograd.AutoContext.get_instance().close_device()
