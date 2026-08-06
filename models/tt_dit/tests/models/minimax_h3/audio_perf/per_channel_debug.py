"""Where does the per-channel-taps merge lose 4.78e-04?

Merging two C=8 filters into one C=16 call disagreed with running them separately. The width is
exonerated (`width16_defect.py`: every width 8-64 is fp32-grade), so the cause is in something the
merge does and the plain path does not. Two candidates, isolated here:

  A. the per-channel weight path -- pass per-channel taps that are all *identical*, which is
     mathematically the broadcast case, so any difference is the weight construction/preparation.
  B. the doubled input -- `concat([x, x], dim=2)` then slice back out, with no conv at all, to check
     the concat/slice round trip is lossless at these narrow widths.

Whichever shows the 4.78e-04 is the culprit; if neither does, it is the interaction and the merge case
itself needs bisecting.
"""

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d, depthwise_tap_filter
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

T_PAD, K = 82806, 7


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, K).tolist()
        for C in (8, 16):
            torch.manual_seed(0)
            x = torch.randn(2, T_PAD, C) * 0.3
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            print(f"\n=== C={C} ===")

            # A: broadcast taps vs per-channel taps carrying the same values
            bcast = ttnn.to_torch(
                depthwise_tap_filter(xd, taps, 1, mesh_device=device, dtype=ttnn.float32, cache={})
            ).float()
            perch = ttnn.to_torch(
                depthwise_tap_filter(xd, [list(taps)] * C, 1, mesh_device=device, dtype=ttnn.float32, cache={})
            ).float()
            same = torch.equal(bcast, perch)
            print(f"  A weight path   identical taps: exact={same}  maxdiff={float((bcast-perch).abs().max()):.3e}")

            # B: concat/slice round trip, no conv
            x2 = ttnn.concat([xd, xd], dim=2)
            back = ttnn.to_torch(ttnn.slice(x2, [0, 0, 0], [2, T_PAD, C])).float()
            print(
                f"  B concat+slice  round trip:     exact={torch.equal(back, x)}  "
                f"maxdiff={float((back - x).abs().max()):.3e}"
            )

            # C: does a conv over the duplicated input reproduce the single-width conv on each half?
            taps2 = [list(taps)] * (2 * C)
            merged = ttnn.to_torch(
                depthwise_tap_filter(x2, taps2, 1, mesh_device=device, dtype=ttnn.float32, cache={})
            ).float()
            lo, hi = merged[:, :, :C], merged[:, :, C:]
            print(
                f"  C merged halves vs bcast:       lo_exact={torch.equal(lo, bcast)} "
                f"hi_exact={torch.equal(hi, bcast)}  "
                f"lo_maxdiff={float((lo - bcast).abs().max()):.3e}"
            )
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
