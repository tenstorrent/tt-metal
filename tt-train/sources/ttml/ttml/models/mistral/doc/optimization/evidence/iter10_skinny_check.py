"""Correctness and timing of ttml.ops.metal.skinny_matmul against ttnn.matmul at the LoRA shapes (one chip)."""
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


for M, K, N, tb in [
    (1024, 5120, 16, True),
    (1024, 8192, 16, True),
    (1024, 1024, 16, True),
    (1024, 5120, 16, False),
    (1024, 1536, 16, False),
]:
    x = dev_t(rng.normal(0, 1, (1, 1, M, K)))
    w_np = rng.normal(0, 0.05, (1, 1, N, K) if tb else (1, 1, K, N))
    w = dev_t(w_np)
    ref = host(ttnn.matmul(x, w, transpose_b=tb, compute_kernel_config=cfg))
    got = host(ttml.ops.metal.skinny_matmul(x, w, tb))
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    t_ttnn = bench(lambda: ttnn.matmul(x, w, transpose_b=tb, compute_kernel_config=cfg))
    t_sk = bench(lambda: ttml.ops.metal.skinny_matmul(x, w, tb))
    print(
        f"M={M} K={K} N={N} transpose_b={tb}: shape {got.shape} pcc {pcc:.6f} rel {err:.2e} | wall us/op ttnn {t_ttnn:.1f} skinny {t_sk:.1f}",
        flush=True,
    )
ttml.autograd.AutoContext.get_instance().close_device()
