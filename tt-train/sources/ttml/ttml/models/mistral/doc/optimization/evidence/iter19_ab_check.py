"""Interleaved A/B timing (5 rounds, min and median of per-round means) of the four big TP base matmuls:
ttnn's default program config, the model's call (core_grid = full grid) and the best swept 2D configs."""
import statistics, time
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


def dev_t(shape):
    return ttml.autograd.Tensor.from_numpy(
        rng.normal(0, 1, shape).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    ).get_value()


def bench(fn, iters=10):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


def pc2d(gx, gy, pcM, pcN, w, sh, sw):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pcM,
        per_core_N=pcN,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


cases = [
    (
        "w2 fwd",
        1024,
        8256,
        5120,
        False,
        {"2D 11x10 w6 s4x1": pc2d(11, 10, 4, 15, 6, 4, 1), "2D 11x8 w6 s4x1": pc2d(11, 8, 4, 15, 6, 4, 1)},
    ),
    ("gate_up fwd", 1024, 5184, 16384, False, {}),
    ("w2 dX", 8256, 5120, 1024, True, {"2D 8x10 w5 s2x2": pc2d(8, 10, 26, 4, 5, 2, 2)}),
    ("gate_up dX", 5184, 16384, 1024, True, {"2D 10x10 w8 s1x4": pc2d(10, 10, 17, 4, 8, 1, 4)}),
    ("qkv fwd", 1024, 5184, 1536, False, {}),
    ("out fwd", 1024, 1088, 5120, False, {}),
]
for name, M, K, N, tb, pcs in cases:
    a = dev_t((1, 1, M, K))
    w = dev_t((1, 1, N, K) if tb else (1, 1, K, N))
    mm = lambda **kw: ttnn.matmul(a, w, transpose_b=tb, compute_kernel_config=cfg, **kw)
    variants = {"default": lambda: mm(), "full core_grid": lambda: mm(core_grid=full)}
    for k, pc in pcs.items():
        variants[k] = lambda pc=pc: mm(program_config=pc)
    times = {k: [] for k in variants}
    for rnd in range(5):
        for k, fn in variants.items():
            times[k].append(bench(fn))
    gf = 2 * M * K * N / 1e12
    print(
        f"{name} [{M},{K}]x[{K},{N}]{' tb' if tb else ''}: "
        + "; ".join(
            f"{k} min {min(v):.0f} med {statistics.median(v):.0f} us ({gf / (statistics.median(v) * 1e-6):.0f} TFLOP/s)"
            for k, v in times.items()
        ),
        flush=True,
    )
    for t in (a, w):
        t.deallocate()
ttml.autograd.AutoContext.get_instance().close_device()
