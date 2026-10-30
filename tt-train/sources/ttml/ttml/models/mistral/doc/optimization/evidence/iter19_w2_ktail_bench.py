"""w2 base matmul at the fused-K shapes: [1024, K] x [K, 5120] (forward) and [1024, 5120] x [5120, K]^T-free dX
([1024, 5120] x [5120, K]) for K = 8192 (no LoRA tail), 8256 (today's tail of 2 tiles: 258 = 2*3*43 K tiles),
8320 (260 = 4*5*13), 8448 (264 = 8*3*11). HiFi4 + fp32 accumulate, ttnn's default program config."""
import time, sys
import numpy as np
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
rng = np.random.default_rng(0)


def dev_t(shape):
    return ttml.autograd.Tensor.from_numpy(
        rng.normal(0, 1, shape).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    ).get_value()


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


for K in [8192, 8256, 8320, 8448, 8704]:
    a = dev_t((1, 1, 1024, K))
    w = dev_t((1, 1, K, 5120))
    t_fw = bench(lambda: ttnn.matmul(a, w, compute_kernel_config=cfg))
    g = dev_t((1, 1, 1024, 5120))
    wt = dev_t((1, 1, 5120, K))
    t_dx = bench(lambda: ttnn.matmul(g, wt, compute_kernel_config=cfg))
    gf = 2 * 1024 * K * 5120 / 1e12
    print(
        f"K={K} ({K // 32} tiles): fwd {t_fw:.0f} us ({gf / (t_fw * 1e-6):.0f} TFLOP/s)  dX [1024,5120]x[5120,K] {t_dx:.0f} us ({gf / (t_dx * 1e-6):.0f} TFLOP/s)",
        flush=True,
    )
    for t in (a, w, g, wt):
        t.deallocate()
ttml.autograd.AutoContext.get_instance().close_device()
