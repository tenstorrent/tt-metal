"""Correctness and wall time of ttml.ops.metal.skinny_contract against ttnn.matmul(transpose_a=True) (+ multiply)."""
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


def reference(small, big, s, transpose_out):
    if transpose_out:  # dB = s * big^T @ small  -> [N, r]
        out = ttnn.matmul(big, small, transpose_a=True, compute_kernel_config=cfg)
    else:  # dA = small^T @ big -> [r, N]
        out = ttnn.matmul(small, big, transpose_a=True, compute_kernel_config=cfg)
    return ttnn.multiply(out, s) if s != 1.0 else out


# (T, r, N, scale, transpose_out): gate_up dB (16384), out/w2 dB (5120), qkv dB (1536), dA at in = 5120 / 8192 / 1024
for T, r, N, s, to in [
    (1024, 16, 16384, 2.0, True),
    (1024, 16, 5120, 2.0, True),
    (1024, 16, 1536, 0.5, True),
    (1024, 16, 5120, 1.0, False),
    (1024, 16, 8192, 1.0, False),
    (1024, 16, 1024, 1.0, False),
    (2048, 32, 2048, 1.0, True),
]:
    small = dev_t(rng.normal(0, 1, (1, 1, T, r)))
    big = dev_t(rng.normal(0, 1, (1, 1, T, N)))
    ref = host(reference(small, big, s, to))
    got = host(ttml.ops.metal.skinny_contract(small, big, s, to))
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    t_ref = bench(lambda: reference(small, big, s, to))
    t_new = bench(lambda: ttml.ops.metal.skinny_contract(small, big, s, to))
    print(
        f"T={T} r={r} N={N} s={s} transpose_out={to}: shape {got.shape} pcc {pcc:.6f} rel {err:.2e} | wall us ttnn {t_ref:.0f} skinny_contract {t_new:.0f}",
        flush=True,
    )
ttml.autograd.AutoContext.get_instance().close_device()
