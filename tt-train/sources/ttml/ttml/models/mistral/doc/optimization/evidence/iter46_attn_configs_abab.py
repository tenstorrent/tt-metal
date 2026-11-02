"""Iteration 46: the four attention-projection base matmuls at the TP=4 per-chip shapes, ttnn's default choice
against the swept 2D config (iter46_sweep2d_*.log), alternating rounds (ABAB x3) on one chip so the clock drift
cancels; max |diff| against the default output. HiFi4 + fp32 accumulate."""
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
grid = dev.compute_with_storage_grid_size()
full = ttnn.CoreGrid(y=grid.y, x=grid.x)
# (name, M, K, N, transpose_b, grid, in0_block_w, sub_h, sub_w, pcM, pcN)
CASES = [
    ("qkv fwd [1024,5120]x[1536,5120]^T", 1024, 5120, 1536, True, (11, 8), 16, 4, 1, 4, 5),
    ("out fwd [1024,1024]x[5120,1024]^T", 1024, 1024, 5120, True, (11, 8), 8, 4, 1, 4, 15),
    ("qkv dX [1024,1536]x[1536,5120]", 1024, 1536, 5120, False, (11, 10), 12, 4, 1, 4, 15),
    ("out dX [1024,5120]x[5120,1024]", 1024, 5120, 1024, False, (11, 8), 16, 2, 1, 4, 3),
]


def dev_t(shape):
    return ttml.autograd.Tensor.from_numpy(
        rng.normal(0, 1, shape).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    ).get_value()


def host(t):
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32)


def bench(fn, iters=20):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


for name, M, K, N, tb, g, ibw, sh, sw, pcM, pcN in CASES:
    a = dev_t((1, 1, M, K))
    w = dev_t((1, 1, N, K) if tb else (1, 1, K, N))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=g,
        in0_block_w=ibw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pcM,
        per_core_N=pcN,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    default = lambda: ttnn.matmul(a, w, transpose_b=tb, compute_kernel_config=cfg, core_grid=full)
    swept = lambda: ttnn.matmul(a, w, transpose_b=tb, compute_kernel_config=cfg, program_config=pc)
    md = float(np.abs(host(default()) - host(swept())).max())
    td, ts = [], []
    for _ in range(3):
        td.append(bench(default))
        ts.append(bench(swept))
    gf = 2 * M * K * N / 1e12
    print(
        f"{name}: default {np.median(td):.0f} us ({gf / (np.median(td) * 1e-6):.0f} TFLOP/s) "
        f"-> swept grid {g} in0_block_w {ibw} sub {sh}x{sw} pc {pcM}x{pcN}: {np.median(ts):.0f} us "
        f"({gf / (np.median(ts) * 1e-6):.0f} TFLOP/s), {100 * (np.median(ts) / np.median(td) - 1):+.0f} %, "
        f"rounds {[round(x) for x in td]} vs {[round(x) for x in ts]}, max|diff| {md:.3g}",
        flush=True,
    )
    for t in (a, w):
        ttnn.deallocate(t)
ttml.autograd.AutoContext.get_instance().close_device()
