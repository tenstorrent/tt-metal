# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The Wormhole vocoder failure, down to one operator and one band of lengths.

On Wormhole, vocoding a known-good mel at some lengths saturates the waveform, and the
per-stage walk (`probe_hift_isolate.py`) finds `src1` --
`source_resblocks[1](source_downs[1](s_stft))`, at upsample stage 1 -- non-finite while
its input and the other branches stay finite. The streaming path reaches that band
because prepending the 20-frame `hift_mel` cache takes chunk 1 from 110 frames to 130.

This drives the two modules directly, without the vocoder, to find:

  - where the band starts and ends, in `stft_frames = 64L + 1` rather than in L;
  - which operator first produces `inf` -- the `k=1` down-projection, one of the six
    dilated convolutions in the ResBlock, or a Snake activation between them;
  - whether Blackhole has the same band, since the same script runs on both.

Input is synthetic but scale-matched to the measured `s_stft` (RMS 0.061, max 0.46). A
finite, well-conditioned input that comes back non-finite is a kernel defect and needs
no model context to report.

    python3 models/demos/cosyvoice/scripts/probe_hift_source_branch.py [--lengths ...]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GOLDEN = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")

# Mel-frame counts. The band found by probe_hift_isolate is 128..132+; this brackets it on
# both sides and steps through it, plus the two streaming lengths (110 works, 130 fails).
DEFAULT = "110,120,124,126,127,128,130,132,134,136,138,140,142,143,144,146,160,172"


def report(t) -> tuple[float, float, int]:
    """RMS (of the finite part), max-abs, and how many elements are not finite."""
    x = ttnn.to_torch(t).float()
    bad = int((~torch.isfinite(x)).sum())
    fin = x[torch.isfinite(x)]
    return (
        float(fin.pow(2).mean().sqrt()) if fin.numel() else float("nan"),
        float(fin.abs().max()) if fin.numel() else float("nan"),
        bad,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default=DEFAULT)
    ap.add_argument("--l1", type=int, default=131072)
    args = ap.parse_args()

    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.weights import WeightBag

    device = ttnn.open_device(device_id=0, l1_small_size=args.l1)
    try:
        hift = TtHiFTGenerator(device, WeightBag.load(os.path.join(GOLDEN, "hift_weights.npz")))
        down, rb = hift.source_downs[1], hift.source_resblocks[1]

        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}")
        print(f"  source_downs[1]: Conv1d(18 -> 128, k=1) then source_resblocks[1] (k=3, dil 1/3/5)")
        print(f"  input: randn * 0.06, matching the measured s_stft scale\n")
        print(f"  {'L':>5}{'stft_frames':>13}{'down rms/max':>20}{'   first non-finite step':<28}{'n_inf':>9}")
        print("  " + "-" * 78)

        for L in [int(x) for x in args.lengths.split(",")]:
            frames = 64 * L + 1
            torch.manual_seed(1986)
            x = ttnn.from_torch(
                torch.randn(1, frames, 18) * 0.06, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            si, _ = down(x, frames, 1)
            d_rms, d_max, d_bad = report(si)

            # Walk the ResBlock by hand rather than calling it, so the first step that
            # goes non-finite is named instead of inferred from the block's output.
            first, n_bad = ("-", 0) if d_bad == 0 else ("source_downs[1] (k=1 conv)", d_bad)
            cur = si
            if d_bad == 0:
                for i in range(rb.n):
                    xt = rb.act1[i](cur)
                    _, _, b = report(xt)
                    if b:
                        first, n_bad = (f"snake act1[{i}] (dil {rb.dilations[i]})", b)
                        ttnn.deallocate(xt)
                        break
                    nx, _ = rb.convs1[i](xt, frames, 1)
                    ttnn.deallocate(xt)
                    _, _, b = report(nx)
                    if b:
                        first, n_bad = (f"convs1[{i}] (k=3, dil {rb.dilations[i]})", b)
                        ttnn.deallocate(nx)
                        break
                    xt = rb.act2[i](nx)
                    ttnn.deallocate(nx)
                    _, _, b = report(xt)
                    if b:
                        first, n_bad = (f"snake act2[{i}]", b)
                        ttnn.deallocate(xt)
                        break
                    nx, _ = rb.convs2[i](xt, frames, 1)
                    ttnn.deallocate(xt)
                    _, _, b = report(nx)
                    if b:
                        first, n_bad = (f"convs2[{i}] (k=3, dil 1)", b)
                        ttnn.deallocate(nx)
                        break
                    nxt = ttnn.add(cur, nx)
                    ttnn.deallocate(nx)
                    if cur is not si:
                        ttnn.deallocate(cur)
                    cur = nxt
            if cur is not si:
                ttnn.deallocate(cur)
            ttnn.deallocate(si)
            ttnn.deallocate(x)

            mark = "" if n_bad == 0 else "  <--"
            print(f"  {L:>5}{frames:>13}{f'{d_rms:8.4f} /{d_max:7.3f}':>20}   {first:<28}{n_bad:>9}{mark}")

        print("\n  The band's edges, in stft_frames, are what an upstream report needs:")
        print("  a finite input of ordinary magnitude must not come back non-finite.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
