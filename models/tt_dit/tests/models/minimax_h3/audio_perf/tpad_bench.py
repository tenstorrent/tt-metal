"""Can `_zero_pad_t`'s concat become a `ttnn.pad`, and what does it save?

PROFILE_2026_08_06.txt puts ConcatDeviceOperation at **285.3 ms over 469 calls, 20.4 % of the decode**
-- the single largest line item, and ~608 us per call against a ~142 us average op. Actual convolution
(Conv3d + Conv2d) is only 292 ms. So the decode is dominated by data movement, and concat leads it.

`_zero_pad_t` builds `concat([zeros, x, zeros], dim=1)`, which is exactly what `ttnn.pad` expresses on
the T axis. The same substitution in `_pad_channels_to_aligned` measured exact and 4-30x faster, so it is worth
checking here at the T-padding shapes the bands actually use, before touching shipping code.

Also times the replicate case, whose right-hand pad is a scaled copy of the last row rather than zeros
-- pad cannot express that directly, so it is measured only to size what remains if zeros are fixed.
"""

import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _persistent_zeros
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

# (label, C, T, pad) -- the tail stages, where most of the 127 bands live
CASES = [
    ("s4 C32", 32, 41403, 11),
    ("s5 C16", 16, 82806, 11),
    ("s6 C8", 8, 165606, 11),
    ("s6up C8", 8, 331212, 11),
]
ITERS = 5


def timed(fn):
    fn()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'case':<10} {'rows':>8} {'concat ms':>10} {'pad ms':>8} {'speedup':>8} {'exact':>7} {'maxdiff':>11}")
        print("-" * 70)
        tot_c = tot_p = 0.0
        for label, C, T, pad in CASES:
            torch.manual_seed(0)
            x = torch.randn(2, T, C) * 0.3
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            B = 2

            def by_concat():
                zl = _persistent_zeros(
                    (B, pad, C), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_device=device
                )
                zr = _persistent_zeros(
                    (B, pad, C), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_device=device
                )
                return ttnn.concat([zl, xd, zr], dim=1)

            def by_pad():
                return ttnn.pad(xd, [(0, 0), (pad, pad), (0, 0)], value=0.0)

            try:
                a, b = by_concat(), by_pad()
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<10} {2 * T:>8}  FAILED {str(exc).splitlines()[0][:40]}")
                continue

            ga, gb = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
            ref = torch.zeros(B, T + 2 * pad, C)
            ref[:, pad : pad + T, :] = x
            if tuple(gb.shape) != tuple(ref.shape):
                print(f"{label:<10} {2 * T:>8}  SHAPE {tuple(gb.shape)} != {tuple(ref.shape)}")
                continue
            d = float((gb - ref).abs().max())
            same = torch.equal(ga, gb)

            tc, tp = timed(by_concat), timed(by_pad)
            tot_c += tc
            tot_p += tp
            print(f"{label:<10} {2 * T:>8} {tc:>10.3f} {tp:>8.3f} {tc / tp:>7.2f}x {str(same):>7} {d:>11.3e}")

        print("-" * 70)
        print(f"{'total':<10} {'':>8} {tot_c:>10.3f} {tot_p:>8.3f} {tot_c / max(tot_p, 1e-9):>7.2f}x")
        print("\n'exact' compares pad against concat; 'maxdiff' is pad against a CPU-built reference.")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
