"""Is the snake paying a tile-padding tax at the narrow deep stages, and can folding T into C fix it?

The audio tail runs at C=8 and C=16. A 32-wide tile therefore carries 8 or 16 useful lanes, so every
TILE-layout op at those stages moves 4x or 2x more bytes than the data warrants. `snake_beta` is
elementwise with per-channel alpha/beta, which means folding F consecutive timesteps into the channel
axis is exactly equivalent as long as alpha/beta are tiled F times -- and it makes the tile full.

Measures, at the production s5/s6 shapes:
  * to_layout(TILE) + snake_beta + to_layout(RM), as `SnakeBeta.forward` does it today
  * the same with T folded into C so the tile is full
  * the folded form's numerics against the unfolded one (must be bit-identical)
"""

import os
import statistics
import time

import torch

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

# (label, C, T) at the production tail. T values are the post-upsample lengths the snake sees.
SHAPES = [
    ("s3 C64  T20701", 64, 20701),
    ("s4 C32  T41403", 32, 41403),
    ("s4 C32  T124206", 32, 124206),
    ("s5 C16  T82806", 16, 82806),
    ("s5 C16  T165606", 16, 165606),
    ("s6 C8   T165606", 8, 165606),
    ("s6up C8 T331212", 8, 331212),
]
ITERS = int(os.environ.get("SNAKE_ITERS", "5"))


def timed(fn, iters=ITERS):
    fn()
    ttnn.synchronize_device(fn.device)
    ts = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        ttnn.synchronize_device(fn.device)
        ts.append((time.perf_counter() - start) * 1e3)
    return statistics.median(ts)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'shape':<18} {'variant':<14} {'ms':>8}  {'note'}")
        print("-" * 62)
        for label, C, T in SHAPES:
            torch.manual_seed(0)
            x = torch.randn(2, T, C) * 0.3
            a = torch.rand(1, 1, C) + 0.5
            b = torch.rand(1, 1, C) + 0.5

            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            ad = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            bd = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

            def plain(_xd=xd, _ad=ad, _bd=bd):
                t = ttnn.to_layout(_xd, ttnn.TILE_LAYOUT)
                y = ttnn.snake_beta(t, _ad, _bd)
                return ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)

            plain.device = device
            ref = ttnn.to_torch(plain()).float()
            print(f"{label:<18} {'today':<14} {timed(plain):>8.2f}")

            from models.tt_dit.layers.audio_ops import SnakeBeta as _SB

            fold = _SB._fold_factor(type("S", (), {"parallel_config": None})(), T, C)
            if fold > 1:
                # Fold `fold` timesteps into C so the tile is full. Elementwise + per-channel means
                # this is an exact re-indexing, provided alpha/beta repeat with the same period.
                xf = x.reshape(2, T // fold, C * fold)
                af = a.repeat(1, 1, fold)
                bf = b.repeat(1, 1, fold)
                xfd = ttnn.from_torch(xf, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                afd = ttnn.from_torch(af, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
                bfd = ttnn.from_torch(bf, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

                def folded(_xd=xfd, _ad=afd, _bd=bfd):
                    t = ttnn.to_layout(_xd, ttnn.TILE_LAYOUT)
                    y = ttnn.snake_beta(t, _ad, _bd)
                    return ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)

                folded.device = device
                got = ttnn.to_torch(folded()).float().reshape(2, T, C)
                exact = torch.equal(got, ref)
                maxdiff = float((got - ref).abs().max())
                ms = timed(folded)
                print(f"{label:<18} {'fold x' + str(fold):<14} {ms:>8.2f}  bit-exact={exact} maxdiff={maxdiff:.2e}")
            print()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
