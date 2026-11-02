"""Iteration 46 scoping: the attention out-projection forward in the transposed orientation (the w2 machinery of
iteration 36) on one chip: today's [1024, 1088] x [5120, 1088]^T on ttnn's full-grid choice (88 cores) against
y^T = W_cat^T [5120, 1088] x x_cat [1024, 1088]^T on 110 cores (per_core_M 16, per_core_N 3), configs swept over
in0_block_w {1, 2} and the subblocks; alternating rounds (ABAB) so the clock drift cancels; max |diff| against
today's output transposed. HiFi4 + fp32 accumulate."""
import sys, time
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
T, K, N = 1024, 1024 + 64, 5120


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


x = dev_t((1, 1, T, K))
w = dev_t((1, 1, N, K))  # W_cat [out, in + tail]
wt = ttnn.transpose(
    w, -2, -1
)  # [in + tail, out]: the plain-orientation operand today is W_cat^T? no: today in1 = W_cat with transpose_b
full = ttnn.CoreGrid(y=grid.y, x=grid.x)


def today():
    return ttnn.matmul(x, w, transpose_b=True, compute_kernel_config=cfg, core_grid=full)


ref = host(today())
pcM, pcN = -(-(N // 32) // grid.y), -(-(T // 32) // grid.x)
gf = 2 * T * K * N / 1e12
cands = []
for in0_block_w in (1, 2):
    for sh, sw in ((1, 3), (2, 3), (4, 1), (2, 1), (1, 1)):
        if pcM % sh or pcN % sw:
            continue
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=pcM,
            per_core_N=pcN,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
        cands.append((f"T grid {grid.x}x{grid.y} pcM {pcM} pcN {pcN} in0_block_w {in0_block_w} sub {sh}x{sw}", pc))


def transposed(pc):
    return ttnn.matmul(w, x, transpose_b=True, compute_kernel_config=cfg, program_config=pc)


results = {}
for name, pc in cands:
    try:
        out = host(transposed(pc))
        md = float(np.abs(out.reshape(N, T).T - ref.reshape(T, N)).max())
        results[name] = (pc, md)
        print(f"{name}: max|diff| vs today {md:.3g}", flush=True)
    except Exception as e:
        print(f"{name}: FAIL {str(e).splitlines()[0][:120]}", flush=True)
times = {n: [] for n in ["today"] + list(results)}
for rnd in range(3):
    for n in ["today"] + list(results):
        t = bench(today) if n == "today" else bench(lambda: transposed(results[n][0]))
        times[n].append(t)
        print(f"round {rnd} {n}: {t:.0f} us ({gf / (t * 1e-6):.0f} TFLOP/s)", flush=True)
# the backward's dX product with the gradient as it arrives: today [1088, 5120] x [1024, 5120]^T (transpose_b),
# transposed route [1088, 5120] x [5120, 1024] plain (the fused residual norm's transposed gradient).
g = dev_t((1, 1, T, N))
g_t = ttnn.transpose(g, -2, -1)
wk = ttnn.transpose(w, -2, -1)  # w_val [in + tail, out]


def dx_today():
    return ttnn.matmul(wk, g, transpose_b=True, compute_kernel_config=cfg, core_grid=full)


def dx_plain():
    return ttnn.matmul(wk, g_t, compute_kernel_config=cfg, core_grid=full)


dmd = float(np.abs(host(dx_today()) - host(dx_plain())).max())
print(f"dX plain route max|diff| vs transpose_b route {dmd:.3g}", flush=True)
for rnd in range(3):
    for n, fn in (("dx_today", dx_today), ("dx_plain", dx_plain)):
        t = bench(fn)
        times.setdefault(n, []).append(t)
        print(f"round {rnd} {n}: {t:.0f} us ({gf / (t * 1e-6):.0f} TFLOP/s)", flush=True)
print("median us per config:")
for n, ts in sorted(times.items(), key=lambda kv: np.median(kv[1])):
    print(f"  {np.median(ts):7.0f}  {n}")
ttml.autograd.AutoContext.get_instance().close_device()
