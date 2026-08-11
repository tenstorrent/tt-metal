"""Is one depthwise conv at 2C cheaper than two at C?

row_cost.py says a row costs ~4.2 ns almost regardless of width, so doubling the channel axis should
be nearly free and merging the band's paired convs should nearly halve their cost. If instead 2C costs
2x C, the merge only saves the ~180 us per-op floor and the extra dup/reduce passes swamp it.
"""
import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import depthwise_tap_filter


def med(fn, dev, iters=5):
    fn()
    ttnn.synchronize_device(dev)
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        s.append((time.perf_counter() - t0) * 1e3)
        if o is not None:
            ttnn.deallocate(o)
    return statistics.median(s)


d = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
K = 7
try:
    print(f"{'shape':<16} {'2 convs @C':>11} {'1 conv @2C':>11} {'ratio':>7}")
    for C, M in [(32, 41403), (16, 82806), (8, 165606)]:
        B = 2
        taps = torch.randn(K).tolist()
        pc = [torch.randn(K).tolist() for _ in range(2 * C)]
        xc = ttnn.from_torch(torch.randn(B, M, C) * 0.3, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=d)
        x2 = ttnn.from_torch(torch.randn(B, M, 2 * C) * 0.3, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=d)
        c1, c2 = {}, {}

        def pair():
            a = depthwise_tap_filter(xc, taps, 1, mesh_device=d, dtype=ttnn.float32, cache=c1)
            b = depthwise_tap_filter(xc, taps, 1, mesh_device=d, dtype=ttnn.float32, cache=c1)
            ttnn.deallocate(a)
            return b

        def wide():
            return depthwise_tap_filter(x2, pc, 1, mesh_device=d, dtype=ttnn.float32, cache=c2)

        tp, tw = med(pair, d), med(wide, d)
        print(f"C={C:<3} M={M:<9} {tp:11.2f} {tw:11.2f} {tw/tp:7.2f}")
        ttnn.deallocate(xc)
        ttnn.deallocate(x2)
finally:
    ttnn.close_mesh_device(d)
