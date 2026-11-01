"""vp_ce_row_stats against numpy at the vocab-parallel CE shard shape [1, 1, 1024, 8192] bf16, and the wall time of
the new path (two ops + two tiny reductions) against the old one (ttnn.max; subtract + exp into fp32; ttnn.sum)."""
import time

import numpy as np
import ttnn
import ttml

metal = ttml.ops.metal
ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
S, V = 1024, 8192


def dev_t(a, dtype=ttnn.DataType.BFLOAT16):
    return ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, dtype).get_value()


def host(t):
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)


def bf16(a):  # round to nearest even, as the device conversion does
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def bench(fn, iters=30):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


x_np = rng.normal(0, 3, (1, 1, S, V))
x = dev_t(x_np)
xb = bf16(x_np[0, 0])
ok = True
# mode 0: row max
part = metal.vp_ce_row_stats(x, 0)
print("partials shape", list(part.shape), flush=True)
mx = ttnn.max(part, 3, keepdim=True)
mx_h = host(mx)[0, 0, :, 0]
ref_max = xb.max(1)
print(f"  row max: max|diff| {np.abs(mx_h - ref_max).max():.3e} (exact expected)", flush=True)
ok &= np.abs(mx_h - ref_max).max() == 0.0
# mode 1: row sum of exp(x - max) with the fp32 max column
part_s = metal.vp_ce_row_stats(x, 1, mx)
sm = ttnn.sum(part_s, 3, keepdim=True)
sm_h = host(sm)[0, 0, :, 0]
ref_sum = np.exp(xb - ref_max[:, None]).sum(1)
rel = np.abs(sm_h - ref_sum) / ref_sum
print(f"  row sum exp: max rel {rel.max():.3e} mean rel {rel.mean():.3e}", flush=True)
ps = host(part_s)[0, 0]
print(
    f"  partial tiles: columns 1..31 max|.| {np.abs(ps.reshape(S, -1, 32)[:, :, 1:]).max():.3e} (0 expected)",
    flush=True,
)
ok &= rel.max() < 2e-3
# the old path's precision for comparison: ttnn.max (bf16) then sub + exp (fp32) + sum
mx_old = ttnn.max(x, 3, keepdim=True)
e_old = ttnn.exp(ttnn.subtract(x, mx_old, dtype=ttnn.DataType.FLOAT32))
sm_old = ttnn.sum(e_old, 3, keepdim=True)
rel_old = np.abs(host(sm_old)[0, 0, :, 0] - ref_sum) / ref_sum
print(f"  old path row sum exp: max rel {rel_old.max():.3e} mean rel {rel_old.mean():.3e}", flush=True)
# timing
t_new = bench(
    lambda: ttnn.sum(
        metal.vp_ce_row_stats(x, 1, ttnn.max(metal.vp_ce_row_stats(x, 0), 3, keepdim=True)), 3, keepdim=True
    )
)
t_new_max = bench(lambda: metal.vp_ce_row_stats(x, 0))
t_new_sum = bench(lambda: metal.vp_ce_row_stats(x, 1, mx))
t_old = bench(
    lambda: ttnn.sum(
        ttnn.exp(ttnn.subtract(x, ttnn.max(x, 3, keepdim=True), dtype=ttnn.DataType.FLOAT32)), 3, keepdim=True
    )
)
t_old_max = bench(lambda: ttnn.max(x, 3, keepdim=True))
print(
    f"us: new max {t_new_max:.0f} + sumexp {t_new_sum:.0f} (chain {t_new:.0f}) | old max {t_old_max:.0f}, chain {t_old:.0f}",
    flush=True,
)
print("CE STATS " + ("PASS" if ok else "FAIL"), flush=True)
ttml.autograd.AutoContext.get_instance().close_device()
