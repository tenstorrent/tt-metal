"""Iteration 50 scoping: the qkv dX^T [5184, 1536] x [1024, 1536]^T (the fused route, 0.79 ms per 4 layers on 80 cores
after iteration 48 against the composite's 0.51) on more configs than the iteration-46 sweep tried: 110 cores
((11, 10), per core 17 x 3) with 1 x 3 subblocks and deeper K blocks; alternating rounds against the committed
config and ttnn's default choice; max |diff| against the default."""
import time
import numpy as np
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
rng = np.random.default_rng(50)
grid = dev.compute_with_storage_grid_size()
full = ttnn.CoreGrid(y=grid.y, x=grid.x)
M, K, N = 5184, 1536, 1024


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


a, w = dev_t((1, 1, M, K)), dev_t((1, 1, N, K))
gf = 2 * M * K * N / 1e12
cands = {"default (full grid)": None}
for name, g, ibw, sh, sw, pcM, pcN in [
    ("committed (10,10) 17x4 k8 1x4", (10, 10), 8, 1, 4, 17, 4),
    ("(11,10) 17x3 k8 1x3", (11, 10), 8, 1, 3, 17, 3),
    ("(11,10) 17x3 k12 1x3", (11, 10), 12, 1, 3, 17, 3),
    ("(11,10) 17x3 k16 1x3", (11, 10), 16, 1, 3, 17, 3),
    ("(11,10) 17x3 k24 1x3", (11, 10), 24, 1, 3, 17, 3),
    ("(11,10) 17x3 k16 1x1", (11, 10), 16, 1, 1, 17, 3),
    ("(11,10) 17x3 k48 1x3", (11, 10), 48, 1, 3, 17, 3),
]:
    cands[name] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
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


def run(pc):
    return ttnn.matmul(
        a,
        w,
        transpose_b=True,
        compute_kernel_config=cfg,
        **({"program_config": pc} if pc is not None else {"core_grid": full}),
    )


ref = host(run(None))
ok = {}
for name, pc in cands.items():
    try:
        md = float(np.abs(host(run(pc)) - ref).max())
        ok[name] = pc
        print(f"{name}: max|diff| vs default {md:.3g}", flush=True)
    except Exception as e:
        print(f"{name}: FAIL {str(e).splitlines()[0][:120]}", flush=True)
times = {n: [] for n in ok}
for rnd in range(3):
    for n, pc in ok.items():
        times[n].append(bench(lambda: run(pc)))
print("median us per config (3 alternating rounds):")
for n, ts in sorted(times.items(), key=lambda kv: np.median(kv[1])):
    print(
        f"  {np.median(ts):7.0f} us ({gf / (np.median(ts) * 1e-6):4.0f} TFLOP/s)  {n}  rounds {[round(x) for x in ts]}"
    )
ttml.autograd.AutoContext.get_instance().close_device()
