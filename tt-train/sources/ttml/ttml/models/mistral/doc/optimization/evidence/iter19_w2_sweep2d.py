"""2D (block-mcast) program-config sweep for the w2 forward base matmul [1024, K] x [K, 5120] at the fused K
(8256 = 258 tiles) and a padded K (8448 = 264 tiles). HiFi4 + fp32 accumulate. Baseline: ttnn's default config."""
import time, itertools, sys
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
print(f"grid {grid.x}x{grid.y}", flush=True)


def dev_t(shape):
    return ttml.autograd.Tensor.from_numpy(
        rng.normal(0, 1, shape).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    ).get_value()


def bench(fn, iters=15):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


M, N = 1024, 5120
Mt, Nt = M // 32, N // 32
for K in [int(k) for k in (sys.argv[1:] or ["8256", "8448"])]:
    Kt = K // 32
    a = dev_t((1, 1, M, K))
    w = dev_t((1, 1, K, N))
    ref = ttnn.matmul(a, w, compute_kernel_config=cfg)
    t0 = bench(lambda: ttnn.matmul(a, w, compute_kernel_config=cfg))
    gf = 2 * M * K * N / 1e12
    print(f"K={K} ({Kt} tiles) default: {t0:.0f} us ({gf / (t0 * 1e-6):.0f} TFLOP/s)", flush=True)
    results = []
    # grids (cols x rows): rows split M, cols split N
    for gx, gy in [(11, 8), (10, 8), (11, 10), (10, 10), (8, 10), (13, 8), (13, 10)]:
        if gx > grid.x or gy > grid.y:
            continue
        per_core_M = -(-Mt // gy)
        per_core_N = -(-Nt // gx)
        for in0_block_w in [d for d in divisors(Kt) if d <= 32]:
            for sh, sw in [(4, 2), (2, 4), (4, 1), (1, 4), (2, 2), (1, 8), (8, 1), (4, 3), (3, 5), (5, 3)]:
                if sh * sw > 8 or per_core_M % sh or per_core_N % sw:
                    continue
                for tm in (False, True):
                    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=in0_block_w,
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                        transpose_mcast=tm,
                        fused_activation=None,
                        fuse_batch=True,
                    )
                    try:
                        out = ttnn.matmul(a, w, program_config=pc, compute_kernel_config=cfg)
                        t = bench(lambda: ttnn.matmul(a, w, program_config=pc, compute_kernel_config=cfg))
                        ok = np.allclose(
                            ttml.autograd.create_tensor(out).to_numpy(ttnn.DataType.FLOAT32),
                            ttml.autograd.create_tensor(ref).to_numpy(ttnn.DataType.FLOAT32),
                            rtol=2e-2,
                            atol=1e-1,
                        )
                        results.append((t, gx, gy, per_core_M, per_core_N, in0_block_w, sh, sw, tm, ok))
                        print(
                            f"  grid {gx}x{gy} pcM {per_core_M} pcN {per_core_N} in0_block_w {in0_block_w:2d} sub {sh}x{sw} tm {int(tm)}: {t:.0f} us ({gf / (t * 1e-6):.0f} TFLOP/s) match {ok}",
                            flush=True,
                        )
                        out.deallocate()
                    except Exception as e:
                        print(
                            f"  grid {gx}x{gy} pcM {per_core_M} pcN {per_core_N} in0_block_w {in0_block_w:2d} sub {sh}x{sw} tm {int(tm)}: FAIL {str(e).splitlines()[0][:120]}",
                            flush=True,
                        )
    results.sort()
    print(f"K={K} best 5:")
    for r in results[:5]:
        print("   ", r)
    for t in (a, w, ref):
        t.deallocate()
ttml.autograd.AutoContext.get_instance().close_device()
