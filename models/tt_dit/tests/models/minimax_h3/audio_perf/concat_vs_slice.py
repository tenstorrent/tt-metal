"""Which half of the channel-axis round trip is lossy at C=8 -- concat or slice?

`per_channel_debug.py` showed `concat([x, x], dim=2)` followed by `slice` back to C loses 9.764e-04 at
C=8 and nothing at C=16, with no arithmetic involved. That measured the pair. This measures each
alone:

  concat  build (2, T, 2C) on device, read it back, compare against torch.cat
  slice   build (2, T, 2C) on device *natively* (never concatenated), slice out [0:C] and [C:2C],
          compare against the torch slices

Whichever disagrees with torch is the defect. Running C=8 (known bad) beside C=16 and C=32 (known
good) shows whether it is width-dependent or general.
"""

import torch

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

T = 82806


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'C':>4}  {'op':<8} {'exact':>7} {'maxdiff':>12}")
        print("-" * 36)
        for C in (8, 16, 32):
            torch.manual_seed(0)
            x = torch.randn(2, T, C) * 0.3
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

            # concat alone: no slice anywhere in the path
            cat_ref = torch.cat([x, x], dim=2)
            cat_got = ttnn.to_torch(ttnn.concat([xd, xd], dim=2)).float()
            d = float((cat_got - cat_ref).abs().max()) if cat_got.shape == cat_ref.shape else float("nan")
            print(f"{C:>4}  {'concat':<8} {str(torch.equal(cat_got, cat_ref)):>7} {d:>12.3e}")

            # slice alone: wide tensor built natively, never concatenated
            wide = torch.randn(2, T, 2 * C) * 0.3
            wd = ttnn.from_torch(wide, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            lo = ttnn.to_torch(ttnn.slice(wd, [0, 0, 0], [2, T, C])).float()
            hi = ttnn.to_torch(ttnn.slice(wd, [0, 0, C], [2, T, 2 * C])).float()
            ok = torch.equal(lo, wide[:, :, :C]) and torch.equal(hi, wide[:, :, C:])
            dd = max(
                float((lo - wide[:, :, :C]).abs().max()),
                float((hi - wide[:, :, C:]).abs().max()),
            )
            print(f"{C:>4}  {'slice':<8} {str(ok):>7} {dd:>12.3e}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
