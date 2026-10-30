"""rmsnorm_bw_allcore (now dispatching to the single-phase rmsnorm_bw_rows) against the two-phase prims and
torch fp64, with wall times, at the model's norm shape."""
import time

import numpy as np
import torch
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)


def bf16(a):
    return torch.tensor(a, dtype=torch.float32).to(torch.bfloat16).to(torch.float64).numpy()


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


for T, C, with_res in [(1024, 5120, True), (1024, 5120, False), (512, 2048, True)]:
    x_np = bf16(rng.normal(0, 1, (1, 1, T, C)))
    g_np = bf16(rng.normal(1, 0.1, (1, 1, 1, C)))
    dy_np = bf16(rng.normal(0, 1, (1, 1, T, C)))
    dres_np = bf16(rng.normal(0, 1, (1, 1, T, C)))
    rms_np = bf16(np.sqrt((x_np**2).mean(-1, keepdims=True) + 1e-6))
    x, g, dy, dres = dev_t(x_np), dev_t(g_np), dev_t(dy_np), dev_t(dres_np)
    rms = dev_t(rms_np)  # [1, 1, T, 1]: column 0 of each tile is what the kernels read
    res = dres if with_res else None
    got = host(ttml.ops.metal.rmsnorm_bw_allcore(x, g, rms, dy, False, res)[0])
    # torch fp64 reference with the same (bf16) rms
    xr, gr, dyr = torch.tensor(x_np), torch.tensor(g_np), torch.tensor(dy_np)
    rr = torch.tensor(rms_np)
    inv = 1.0 / rr
    tt = (gr * dyr * xr).sum(-1, keepdim=True) * inv * inv / C
    ref = (inv * (gr * dyr - tt * xr)).numpy() + (dres_np if with_res else 0.0)
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    t_new = bench(lambda: ttml.ops.metal.rmsnorm_bw_allcore(x, g, rms, dy, False, res))
    print(f"T={T} C={C} residual={with_res}: pcc {pcc:.6f} rel {rel:.2e} | wall us {t_new:.0f}", flush=True)
ttml.autograd.AutoContext.get_instance().close_device()
