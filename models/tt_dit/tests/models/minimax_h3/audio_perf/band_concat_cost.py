"""How much of the fused band is the p0/p1 replicate-pad concats?

The plan to remove them is only worth the disruption if they are actually expensive. Time the exact
call the band makes -- two 6-piece concats over an M-row phase -- against the down convs they feed and
against a whole band, at the production tail shapes.
"""

import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import depthwise_tap_filter

SHAPES = [("s4 C32", 32, 41403), ("s5 C16", 16, 82806), ("s6 C8", 8, 165606)]
ITERS = 5
K = 12


def med(fn):
    fn()
    ttnn.synchronize_device(fn.device)
    s = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(fn.device)
        s.append((time.perf_counter() - t0) * 1e3)
        if out is not None:
            ttnn.deallocate(out)
    return statistics.median(s)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        taps = torch.randn(K).tolist()
        even, odd = [taps[2 * a] for a in range(K // 2)], [taps[2 * a + 1] for a in range(K // 2)]
        print(f"{'shape':<10} {'M':>8} {'2x concat ms':>13} {'2x downconv ms':>15} {'concat share':>13}")
        print("-" * 64)
        for label, C, T in SHAPES:
            # M is the phase length the band actually reaches: T + 2*(pad-crop) - K_sub + 1 with the
            # ratio-2 polyphase sub-taps, i.e. essentially T.
            M = T
            cache = {}
            s0 = ttnn.from_torch(
                torch.randn(2, M, C) * 0.3, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            s1 = ttnn.from_torch(
                torch.randn(2, M, C) * 0.3, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            B = 2
            l0, l1, r0, r1 = 3, 2, 2, 3

            def build():
                first = ttnn.slice(s0, [0, 0, 0], [B, 1, C])
                last = ttnn.slice(s1, [0, M - 1, 0], [B, M, C])
                p0 = ttnn.concat([first] * l0 + [s1] + [last] * r0, dim=1)
                p1 = ttnn.concat([first] * l1 + [s0] + [last] * r1, dim=1)
                ttnn.deallocate(first)
                ttnn.deallocate(last)
                ttnn.deallocate(p1)
                return p0

            build.device = device
            t_concat = med(build)

            first = ttnn.slice(s0, [0, 0, 0], [B, 1, C])
            last = ttnn.slice(s1, [0, M - 1, 0], [B, M, C])
            p0 = ttnn.concat([first] * l0 + [s1] + [last] * r0, dim=1)
            p1 = ttnn.concat([first] * l1 + [s0] + [last] * r1, dim=1)

            def convs():
                a = depthwise_tap_filter(p0, even, 1, mesh_device=device, dtype=ttnn.float32, cache=cache)
                b = depthwise_tap_filter(p1, odd, 1, mesh_device=device, dtype=ttnn.float32, cache=cache)
                out = ttnn.add(a, b)
                ttnn.deallocate(a)
                ttnn.deallocate(b)
                return out

            convs.device = device
            t_conv = med(convs)
            print(f"{label:<10} {M:8d} {t_concat:13.2f} {t_conv:15.2f} {t_concat/(t_concat+t_conv)*100:12.1f}%")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
