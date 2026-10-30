"""Correctness and wall time of ttml.ops.metal.lowrank_update against ttnn matmul + addalpha at the LoRA shapes."""
import time

import numpy as np
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
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


# (images, M, N, K, scale, transpose_b): fused dx += dh @ A (gate_up in=5120, w2 in=8192), composite base + s h B^T
for B, M, N, K, s, tb in [
    (1, 1024, 5120, 16, 1.0, False),
    (1, 1024, 8192, 16, 1.0, False),
    (1, 1024, 1536, 16, 2.0, True),
    (1, 1024, 5120, 16, 2.0, True),
    (2, 512, 1024, 32, 0.5, True),
]:
    acc = dev_t(rng.normal(0, 1, (1, B, M, N)))
    a = dev_t(rng.normal(0, 1, (1, B, M, K)))
    w = dev_t(rng.normal(0, 0.05, (1, 1, N, K) if tb else (1, 1, K, N)))
    ref = host(ttnn.addalpha(acc, ttnn.matmul(a, w, transpose_b=tb, compute_kernel_config=cfg), s))
    got = host(ttml.ops.metal.lowrank_update(acc, a, w, s, tb))
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    t_ref = bench(lambda: ttnn.addalpha(acc, ttnn.matmul(a, w, transpose_b=tb, compute_kernel_config=cfg), s))
    t_new = bench(lambda: ttml.ops.metal.lowrank_update(acc, a, w, s, tb))
    print(
        f"B={B} M={M} N={N} K={K} s={s} transpose_b={tb}: pcc {pcc:.6f} rel {err:.2e} max|diff| {np.abs(got - ref).max():.2e} "
        f"| wall us ttnn matmul+add {t_ref:.0f} lowrank_update {t_new:.0f}",
        flush=True,
    )
ttml.autograd.AutoContext.get_instance().close_device()
