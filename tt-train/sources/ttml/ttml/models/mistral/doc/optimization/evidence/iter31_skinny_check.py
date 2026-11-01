"""skinny_matmul at the model's three shapes: outputs against ttnn.matmul (fp32 accumulate) and wall time per call."""
import time

import numpy as np
import ttnn
import ttml

metal = ttml.ops.metal
ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T, R, TAIL = 1024, 16, 64


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


def report(name, got, ref):
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    diff = np.abs(got - ref)
    print(
        f"  {name}: pcc {pcc:.8f} max|diff| {diff.max():.3e} differing {int((diff > 0).sum())}/{got.size}", flush=True
    )


cfg = ttml.modules.frozen_linear._compute_config()
# (a) x [T, 5120] @ A^T, A [R, 5120]  (gate_up / qkv / out: transpose_b)
x = dev_t(rng.normal(0, 1, (1, 1, T, 5120)))
A = dev_t(rng.normal(0, 0.05, (1, 1, R, 5120)))
h = metal.skinny_matmul(x, A, True)
ref = ttnn.matmul(x, ttnn.transpose(A, -2, -1), compute_kernel_config=cfg)
report("K=5120 transpose_b", host(h)[..., :R], host(ref))
t_a = bench(lambda: metal.skinny_matmul(x, A, True))
# (b) the wide w2 input [T, 8192 + 64] with the tail write
xw = dev_t(rng.normal(0, 1, (1, 1, T, 8192 + TAIL)))
Aw = dev_t(rng.normal(0, 0.05, (1, 1, R, 8192)))
hw = metal.skinny_matmul(xw, Aw, True, xw, 8192 // 32, TAIL // 32)
refw = ttnn.matmul(ttnn.slice(xw, [0, 0, 0, 0], [1, 1, T, 8192]), ttnn.transpose(Aw, -2, -1), compute_kernel_config=cfg)
report("K=8192 wide tail", host(hw)[..., :R], host(refw))
t_b = bench(lambda: metal.skinny_matmul(xw, Aw, True, xw, 8192 // 32, TAIL // 32))
# (c) g [T, 5120] @ B, B [5120, R]  (no transpose)
g = dev_t(rng.normal(0, 1, (1, 1, T, 5120)))
B = dev_t(rng.normal(0, 0.05, (1, 1, 5120, R)))
hb = metal.skinny_matmul(g, B, False)
refb = ttnn.matmul(g, B, compute_kernel_config=cfg)
report("K=5120 plain", host(hb)[..., :R], host(refb))
t_c = bench(lambda: metal.skinny_matmul(g, B, False))
print(f"us per call: K=5120 transpose_b {t_a:.0f} | K=8192 wide tail {t_b:.0f} | K=5120 plain {t_c:.0f}", flush=True)
ttml.autograd.AutoContext.get_instance().close_device()
