"""Can two C-stacked conv outputs be summed cheaply?

Merging the band's paired convs into one at 2C only wins if folding the halves back to C costs less
than the conv invocation it saves (~1.4-3.4 ms at these shapes). Two candidates.
"""
import statistics
import time

import torch

import ttnn


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
try:
    print(f"{'shape':<14} {'sum(dim=2) ms':>14} {'slice+add ms':>13}")
    for C, M in [(32, 41403), (16, 82806), (8, 165606)]:
        B = 2
        x = ttnn.from_torch(torch.randn(B, M, 2 * C) * 0.3, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=d)

        def by_sum():
            v = ttnn.reshape(x, (B, M, 2, C))
            return ttnn.sum(v, dim=2)

        def by_slice():
            a = ttnn.slice(x, [0, 0, 0], [B, M, C])
            b = ttnn.slice(x, [0, 0, C], [B, M, 2 * C])
            r = ttnn.add(a, b)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            return r

        try:
            t_sum = f"{med(by_sum, d):14.2f}"
        except Exception as e:
            t_sum = f"{type(e).__name__[:13]:>14}"
        try:
            t_sl = f"{med(by_slice, d):13.2f}"
        except Exception as e:
            t_sl = f"{type(e).__name__[:12]:>13}"
        print(f"C={C:<3} M={M:<8} {t_sum} {t_sl}")
        ttnn.deallocate(x)
finally:
    ttnn.close_mesh_device(d)
