"""dB = s * big^T @ small ([N, r]) three ways: ttnn transposed-a matmul in the [N, r] orientation (the current
reference), ttnn's fast [r, N] orientation plus a transpose of the small output, and skinny_contract."""
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


def ref_nr(small, big, s):
    out = ttnn.matmul(big, small, transpose_a=True, compute_kernel_config=cfg)
    return ttnn.multiply(out, s) if s != 1.0 else out


def ref_rn_t(small, big, s):
    out = ttnn.matmul(small, big, transpose_a=True, compute_kernel_config=cfg)
    out = ttnn.multiply(out, s) if s != 1.0 else out
    return ttnn.transpose(out, -2, -1)


for T, r, N, s in [(1024, 16, 16384, 2.0), (1024, 16, 5120, 2.0), (1024, 16, 1536, 0.5)]:
    small = dev_t(rng.normal(0, 1, (1, 1, T, r)))
    big = dev_t(rng.normal(0, 1, (1, 1, T, N)))
    a = host(ref_nr(small, big, s))
    b = host(ref_rn_t(small, big, s))
    c = host(ttml.ops.metal.skinny_contract(small, big, s, True))
    print(
        f"T={T} r={r} N={N}: rel(rn_t) {np.linalg.norm(b - a) / np.linalg.norm(a):.2e} rel(op) {np.linalg.norm(c - a) / np.linalg.norm(a):.2e}"
        f" | wall us ttnn[N,r] {bench(lambda: ref_nr(small, big, s)):.0f} ttnn[r,N]+T {bench(lambda: ref_rn_t(small, big, s)):.0f}"
        f" skinny_contract {bench(lambda: ttml.ops.metal.skinny_contract(small, big, s, True)):.0f}",
        flush=True,
    )
ttml.autograd.AutoContext.get_instance().close_device()
