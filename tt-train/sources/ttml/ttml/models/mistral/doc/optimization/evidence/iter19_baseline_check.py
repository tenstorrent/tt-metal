"""The four big TP base matmuls: ttnn's default, the model's call (core_grid = full grid) and the best swept 2D
configs, wall us (HiFi4 + fp32 accumulate), max |diff| vs the model's call."""
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


def dev_t(shape):
    return ttml.autograd.Tensor.from_numpy(
        rng.normal(0, 1, shape).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    ).get_value()


def host(t):
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32)


def bench(fn, iters=15):
    for _ in range(3):
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
        [pc2d(11, 8, 4, 15, 6, 4, 1), pc2d(11, 10, 4, 15, 6, 4, 1), pc2d(11, 8, 4, 15, 3, 4, 1)],
    ),
    ("gate_up fwd", 1024, 5184, 16384, False, [pc2d(11, 8, 4, 47, 1, 4, 1)]),
    ("w2 dX", 8256, 5120, 1024, True, [pc2d(8, 10, 26, 4, 5, 2, 2), pc2d(10, 10, 26, 4, 5, 2, 2)]),
    ("gate_up dX", 5184, 16384, 1024, True, [pc2d(10, 10, 17, 4, 8, 1, 4), pc2d(10, 10, 17, 4, 4, 1, 4)]),
]
for name, M, K, N, tb, pcs in cases:
    a = dev_t((1, 1, M, K))
    w = dev_t((1, 1, N, K) if tb else (1, 1, K, N))
    mm = lambda **kw: ttnn.matmul(a, w, transpose_b=tb, compute_kernel_config=cfg, **kw)
    ref = host(mm(core_grid=full))
    gf = 2 * M * K * N / 1e12
    line = f"{name} [{M},{K}]x[{K},{N}]{' tb' if tb else ''}: default {bench(lambda: mm()):.0f} us, full core_grid {bench(lambda: mm(core_grid=full)):.0f} us"
    for pc in pcs:
        try:
            t = bench(lambda: mm(program_config=pc))
            md = float(np.abs(host(mm(program_config=pc)) - ref).max())
            line += f", 2D[{pc.compute_with_storage_grid_size.x}x{pc.compute_with_storage_grid_size.y} w{pc.in0_block_w} s{pc.out_subblock_h}x{pc.out_subblock_w}] {t:.0f} us (diff {md:g})"
        except Exception as e:
            line += f", 2D FAIL {str(e).splitlines()[0][:60]}"
    print(line, flush=True)
    for t in (a, w):
        t.deallocate()
ttml.autograd.AutoContext.get_instance().close_device()
