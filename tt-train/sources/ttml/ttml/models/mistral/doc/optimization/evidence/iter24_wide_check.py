"""Iteration 24: the wide fused-LoRA input path (w2) piece by piece against the concat path, bit for bit, plus the
cost of the concat + pad it removes. Shapes: the TP=4 w2 shard (T=1024, in=8192, out=5120, r=16)."""
import time

import numpy as np
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T, I, N, R, TAIL = 1024, 8192, 5120, 16, 64
metal = ttml.ops.metal


def dev_t(a):
    return ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16).get_value()


def host(t):
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32)


def same(name, a, b):
    d = np.abs(a - b)
    print(f"    {name}: max|diff| {d.max():.3e} differing {(d > 0).sum()}/{a.size}", flush=True)
    return d.max() == 0.0


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


packed = dev_t(rng.normal(0, 1, (1, 1, T, 2 * I)))
A = dev_t(rng.normal(0, 0.05, (1, 1, R, I)))
Wcat = dev_t(rng.normal(0, 0.02, (1, 1, I + TAIL, N)))
dcat_t = dev_t(rng.normal(0, 1, (1, 1, I + TAIL, T)))
ok = True

print("forward", flush=True)
h_ref = metal.swiglu_packed_fw(packed)
h_wide = metal.swiglu_packed_fw(packed, None, I + TAIL)
print("  swiglu wide", list(h_wide.shape), flush=True)
hw_np = host(h_wide)
ok &= same("swiglu out (first I cols)", hw_np[..., :I], host(h_ref))
hw = metal.skinny_matmul(h_wide, A, True, h_wide, I // 32, TAIL // 32)
hr = metal.skinny_matmul(h_ref, A, True)
ok &= same("h", host(hw), host(hr))
hw_np = host(h_wide)
ok &= same("tail h", hw_np[..., I : I + R], host(hw))
ok &= same("tail zeros", hw_np[..., I + R : I + TAIL], np.zeros((1, 1, T, TAIL - R), np.float32))
ok &= same("x part untouched", hw_np[..., :I], host(h_ref))
cfg = ttml.modules.frozen_linear._compute_config()
h_pad = ttnn.pad(hr, padding=[(0, 0), (0, 0), (0, 0), (0, TAIL - R)], value=0.0)
x_cat = ttnn.concat([h_ref, h_pad], dim=3)
y_ref = ttnn.matmul(x_cat, Wcat, compute_kernel_config=cfg)
y_wide = ttnn.matmul(h_wide, Wcat, compute_kernel_config=cfg)
ok &= same("y", host(y_wide), host(y_ref))

print("backward", flush=True)
dx_r, tail_r = metal.transpose_split(dcat_t, I, R, 2.0)
dx_w, tail_w = metal.transpose_split(dcat_t, I, R, 2.0, I + TAIL)
print("  dx wide", list(dx_w.shape), flush=True)
dxw_np = host(dx_w)
ok &= same("dx (first I cols)", dxw_np[..., :I], host(dx_r))
ok &= same("s_dh", host(tail_w), host(tail_r))
ok &= same("dx tail = unscaled tail rows^T", dxw_np[..., I:], np.swapaxes(host(dcat_t)[..., I:, :], -1, -2))
up_r = metal.lowrank_update(dx_r, tail_r, A)
up_w = metal.lowrank_update(dx_w, tail_w, A)
ok &= same("lowrank_update (first I cols)", host(up_w)[..., :I], host(up_r))
g_r = metal.swiglu_packed_bw(packed, up_r)
g_w = metal.swiglu_packed_bw(packed, up_w)
ok &= same("swiglu bw from the wide grad", host(g_w), host(g_r))

print("cost", flush=True)
t_concat = bench(
    lambda: ttnn.concat([h_ref, ttnn.pad(hr, padding=[(0, 0), (0, 0), (0, 0), (0, TAIL - R)], value=0.0)], dim=3)
)
t_skinny_old = bench(lambda: metal.skinny_matmul(h_ref, A, True))
t_skinny_tail = bench(lambda: metal.skinny_matmul(h_wide, A, True, h_wide, I // 32, TAIL // 32))
t_swiglu = bench(lambda: metal.swiglu_packed_fw(packed))
t_swiglu_wide = bench(lambda: metal.swiglu_packed_fw(packed, None, I + TAIL))
t_ts = bench(lambda: metal.transpose_split(dcat_t, I, R, 2.0))
t_ts_wide = bench(lambda: metal.transpose_split(dcat_t, I, R, 2.0, I + TAIL))
print(
    f"  us: pad+concat {t_concat:.0f} | skinny {t_skinny_old:.0f} -> tail-write {t_skinny_tail:.0f} | swiglu "
    f"{t_swiglu:.0f} -> wide {t_swiglu_wide:.0f} | transpose_split {t_ts:.0f} -> wide {t_ts_wide:.0f}",
    flush=True,
)
print("WIDE PATH", "PASS" if ok else "FAIL", flush=True)
ttml.autograd.AutoContext.get_instance().close_device()
