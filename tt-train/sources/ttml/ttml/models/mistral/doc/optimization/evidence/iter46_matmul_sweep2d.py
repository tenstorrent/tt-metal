"""2D (block-mcast) program-config sweep for one base-matmul shape [M, K] x [K, N] (HiFi4 + fp32 accumulate).
Prints ttnn's default time, every config's time and max |diff| vs the default output, and the best five.
Usage: iter19_matmul_sweep2d.py M K N [--quick] [--tb]   (--tb: the second operand is [N, K], matmul with transpose_b=True)"""
import sys, time
import numpy as np
import ttnn
import ttml

M, K, N = (int(x) for x in sys.argv[1:4])
quick = "--quick" in sys.argv
tb = "--tb" in sys.argv
ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
rng = np.random.default_rng(0)
grid = dev.compute_with_storage_grid_size()


def dev_t(shape):
    return ttml.autograd.Tensor.from_numpy(
        rng.normal(0, 1, shape).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    ).get_value()


def host(t):
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32)


def bench(fn, iters=10):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


Mt, Kt, Nt = M // 32, K // 32, N // 32
a = dev_t((1, 1, M, K))
w = dev_t((1, 1, N, K) if tb else (1, 1, K, N))


def mm(**kw):
    return ttnn.matmul(a, w, transpose_b=tb, compute_kernel_config=cfg, **kw)


ref = mm()
ref_h = host(ref)
t0 = bench(lambda: mm())
gf = 2 * M * K * N / 1e12
print(
    f"[{M},{K}]x[{K},{N}]{' (transpose_b)' if tb else ''} default: {t0:.0f} us ({gf / (t0 * 1e-6):.0f} TFLOP/s)",
    flush=True,
)
results = []
only_tb = tb
grids = (
    [(11, 8), (11, 10), (10, 8), (10, 10), (8, 10), (4, 10), (2, 10), (1, 10)]
    if quick
    else [(11, 8), (10, 8), (11, 10), (10, 10), (8, 10), (9, 10), (6, 10)]
)
subs = [(4, 1), (2, 2), (4, 2), (1, 4)] if quick else [(4, 1), (2, 2), (4, 2), (2, 4), (1, 4), (1, 8), (2, 1), (1, 2)]
for gx, gy in grids:
    if gx > grid.x or gy > grid.y:
        continue
    pcM, pcN = -(-Mt // gy), -(-Nt // gx)
    for in0_block_w in [d for d in range(1, 33) if Kt % d == 0]:
        if in0_block_w * (pcM + pcN) * 2048 * 2 > 1_000_000:  # in0 + in1 blocks, double-buffered
            continue
        for sh, sw in subs:
            if sh * sw > 8 or pcM % sh or pcN % sw:
                continue
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=in0_block_w,
                out_subblock_h=sh,
                out_subblock_w=sw,
                per_core_M=pcM,
                per_core_N=pcN,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
            try:
                out = mm(program_config=pc)
                t = bench(lambda: mm(program_config=pc))
                md = float(np.abs(host(out) - ref_h).max())
                results.append((t, gx, gy, pcM, pcN, in0_block_w, sh, sw, md))
                print(
                    f"  grid {gx}x{gy} pcM {pcM} pcN {pcN} in0_block_w {in0_block_w:2d} sub {sh}x{sw}: {t:.0f} us ({gf / (t * 1e-6):.0f} TFLOP/s) max|diff| {md:.3g}",
                    flush=True,
                )
                out.deallocate()
            except Exception as e:
                print(
                    f"  grid {gx}x{gy} pcM {pcM} pcN {pcN} in0_block_w {in0_block_w:2d} sub {sh}x{sw}: FAIL {str(e).splitlines()[0][:100]}",
                    flush=True,
                )
results.sort()
print(f"[{M},{K}]x[{K},{N}] default {t0:.0f} us; best 5 (us, gx, gy, pcM, pcN, in0_block_w, sub_h, sub_w, max|diff|):")
for r in results[:5]:
    print("   ", r)
ttml.autograd.AutoContext.get_instance().close_device()
