"""Correctness and wall time of ttml.ops.metal.vp_cross_entropy_bw (one chip = one shard, first_v = 0) against
a numpy fp64 reference: scale * g * (exp(x - logZ) - onehot(target))."""
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


def host(t):
    return ttnn.to_torch(t).to(torch.float64).numpy()


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


for S, V, g_val, scale in [(1024, 32768, 1.0, 1.0 / 1024), (1024, 4096, 0.5, 1.0), (256, 2048, 1.0, 1.0 / 256)]:
    x_np = bf16(rng.normal(0, 2, (1, 1, S, V)))
    targets = rng.integers(0, V, (1, S)).astype(np.int64)
    logz = np.log(np.exp(x_np - x_np.max(-1, keepdims=True)).sum(-1, keepdims=True)) + x_np.max(-1, keepdims=True)
    onehot = np.zeros((1, 1, S, V))
    onehot[0, 0, np.arange(S), targets[0]] = 1.0
    ref = scale * g_val * (np.exp(x_np - logz) - onehot)
    x = ttnn.from_torch(torch.tensor(x_np, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    lz = ttnn.from_torch(torch.tensor(logz, dtype=torch.float32), layout=ttnn.TILE_LAYOUT, device=dev)
    tg = ttnn.from_torch(
        torch.tensor(targets, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )
    g = ttnn.from_torch(
        torch.full((1, 1, 32, 32), g_val, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev
    )  # one tile of g
    got = host(ttml.ops.metal.vp_cross_entropy_bw(x, lz, tg, g, V, None, 0, scale))
    diff = np.abs(got - ref)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    rows = np.arange(S)
    tgt_diff = np.abs(got[0, 0, rows, targets[0]] - ref[0, 0, rows, targets[0]]).max()
    t_new = bench(lambda: ttml.ops.metal.vp_cross_entropy_bw(x, lz, tg, g, V, None, 0, scale))
    print(
        f"S={S} V={V} g={g_val} scale={scale:.2e}: pcc {pcc:.6f} rel {rel:.2e} max|diff| {diff.max():.2e} at targets {tgt_diff:.2e} | wall us {t_new:.0f}",
        flush=True,
    )
ttml.autograd.AutoContext.get_instance().close_device()
