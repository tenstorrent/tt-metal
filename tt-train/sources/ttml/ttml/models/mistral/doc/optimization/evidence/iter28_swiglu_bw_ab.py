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


# Review round 10 hard-check gap: swiglu_packed_bw with the narrow [T, I] gradient operand against the wide
# [T, I + 64] one (same bytes read, row stride 8192 -> 8256 columns): the iteration-24 step profile showed
# 1.096 -> 0.745 ms per 4 layers; standalone A/B here.
packed = dev_t(rng.normal(0, 1, (1, 1, T, 2 * I)))
g_narrow = dev_t(rng.normal(0, 1, (1, 1, T, I)))
g_wide = dev_t(rng.normal(0, 1, (1, 1, T, I + TAIL)))
g_wide2 = dev_t(rng.normal(0, 1, (1, 1, T, I + 128)))
for name, g in (("narrow I", g_narrow), ("wide I+64", g_wide), ("wide I+128", g_wide2), ("narrow I again", g_narrow)):
    t = bench(lambda: metal.swiglu_packed_bw(packed, g))
    print(f"swiglu_packed_bw grad {name} {list(g.shape)}: {t:.0f} us", flush=True)
fw_n = bench(lambda: metal.swiglu_packed_fw(packed))
fw_w = bench(lambda: metal.swiglu_packed_fw(packed, None, I + TAIL))
print(f"swiglu_packed_fw narrow {fw_n:.0f} us, wide {fw_w:.0f} us", flush=True)
ttml.autograd.AutoContext.get_instance().close_device()
