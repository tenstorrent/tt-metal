"""Pick a lossless replacement for the concat inside `_pad_channels_to_aligned`.

Root cause established: `ttnn.transpose(-2,-1)` on fp32 ROW_MAJOR truncates to TF32 (`x & 0xFFFFE000`)
at every shape, and `ttnn.concat(dim=-1)` inherits it whenever the row is not a multiple of the
buffer alignment (64B on Blackhole) -- so C=8 and C=24 corrupt, C=16 and C=32 do not.

Candidates for zero-padding C up to 32 without going through that path, scored on exactness first and
speed second (this runs in the decode's inner loop, ~127 bands):

  concat        the status quo, as a baseline for both columns
  pad           ttnn.pad on the last dim -- no concat, no transpose
  tile_concat   to TILE, concat there (TILE fp32 transpose is exact), back to ROW_MAJOR
"""

import time

import torch

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

T = 20701
ALIGN = 32


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'C':>4} {'method':<12} {'exact':>7} {'maxdiff':>12} {'ms':>8}")
        print("-" * 48)
        for C in (8, 16, 24):
            torch.manual_seed(0)
            x = torch.randn(2, T, C) * 0.3
            ref = torch.zeros(2, T, ALIGN)
            ref[:, :, :C] = x
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            pad_c = ALIGN - C

            def m_concat():
                z = ttnn.from_torch(
                    torch.zeros(2, T, pad_c), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                )
                return ttnn.concat([xd, z], dim=2)

            def m_pad():
                return ttnn.pad(xd, [(0, 0), (0, 0), (0, pad_c)], value=0.0)

            def m_tile_concat():
                xt = ttnn.to_layout(xd, ttnn.TILE_LAYOUT)
                z = ttnn.from_torch(
                    torch.zeros(2, T, pad_c), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
                )
                return ttnn.to_layout(ttnn.concat([xt, z], dim=2), ttnn.ROW_MAJOR_LAYOUT)

            for name, fn in (("concat", m_concat), ("pad", m_pad), ("tile_concat", m_tile_concat)):
                try:
                    out = ttnn.to_torch(fn()).float()
                except Exception as exc:  # noqa: BLE001
                    print(f"{C:>4} {name:<12}   FAILED {str(exc).splitlines()[0][:30]}")
                    continue
                if tuple(out.shape) != tuple(ref.shape):
                    print(f"{C:>4} {name:<12}   SHAPE {tuple(out.shape)} != {tuple(ref.shape)}")
                    continue
                d = float((out - ref).abs().max())
                t0 = time.perf_counter()
                for _ in range(5):
                    fn()
                ms = (time.perf_counter() - t0) / 5 * 1e3
                print(f"{C:>4} {name:<12} {str(torch.equal(out, ref)):>7} {d:>12.3e} {ms:>8.2f}")
            print()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
