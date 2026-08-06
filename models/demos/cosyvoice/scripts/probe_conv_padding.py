# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Does the failing `ttnn.conv1d` read its input's tile padding?

`probe_snake_inf.py` moved the blame off Snake and onto the convolution before it:

    source_downs[1] out  max 0.8516
    act1[0] out          max 1.461        <- the conv's input
    convs1[0] out        max 1.582e+38    <- the conv's output

A `k=3` convolution cannot turn a max-1.46 input into 1.6e38 by arithmetic. Two further
results say what kind of fault it is:

  - `COSYVOICE_FIDELITY=HiFi3` does not fix it, so the "HiFi4 + fp32 accumulation is buggy
    on Wormhole" warning tt-metal prints is **not** the cause;
  - `COSYVOICE_FP32_ACC=0` does not fix it either -- it *moves* the band, from
    `stft_frames` 8193-8577 to 8705 and 9217. At `HiFi2`, length 8193 fails with **two**
    bad elements.

Two bad elements is not an overflow. Arithmetic that overflows does so across a region;
two isolated infinities in a million-element tensor is a **read of memory that was never
written**, and "which bytes" changing with the compute config is exactly what that looks
like.

The natural candidate is the input's tile padding. `stft_frames = 64L + 1` is always
`1 (mod 32)`, so every one of these tensors carries 31 padding rows past its logical end,
and a `k=3` convolution computing the last output row reads one row beyond it. If those
rows hold whatever the allocator last left there, and the "same" padding is not being
applied as zeros, the result is precisely this.

Three arms separate the mechanism from the symptom:

    A  as-is                    the failing path
    B  input round-tripped through the host    `from_torch` zero-fills padding by
                                               construction, so if B is clean the
                                               padding is the carrier
    C  input re-tiled on device (row-major and back)   the same fix without a host
                                               round trip -- i.e. an affordable one

    python3 models/demos/cosyvoice/scripts/probe_conv_padding.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GOLDEN = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")

DEFAULT = "8129,8193,8321,8577,8705"


def summarise(t):
    v = ttnn.to_torch(t).float()
    fin = v[torch.isfinite(v)]
    return (
        int((~torch.isfinite(v)).sum()),
        float(fin.abs().max()) if fin.numel() else float("nan"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=DEFAULT)
    ap.add_argument("--l1", type=int, default=131072)
    args = ap.parse_args()

    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.weights import WeightBag

    device = ttnn.open_device(device_id=0, l1_small_size=args.l1)
    try:
        hift = TtHiFTGenerator(device, WeightBag.load(os.path.join(GOLDEN, "hift_weights.npz")))
        down, rb = hift.source_downs[1], hift.source_resblocks[1]
        conv = rb.convs1[0]

        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}")
        print(f"  convs1[0] = Conv1d(128 -> 128, k=3, dilation 1), 'same' padding\n")
        print(
            f"  {'frames':>8}{'pad rows':>10}{'input max':>12}"
            f"{'A as-is':>22}{'B host round-trip':>22}{'C device re-tile':>22}"
        )
        print("  " + "-" * 96)

        for frames in [int(x) for x in args.frames.split(",")]:
            torch.manual_seed(1986)
            x = ttnn.from_torch(
                torch.randn(1, frames, 18) * 0.06, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            si, _ = down(x, frames, 1)
            a1 = rb.act1[0](si)
            in_max = summarise(a1)[1]

            cells = []
            # A: the shipped path.
            out, _ = conv(a1, frames, 1)
            cells.append(summarise(out))
            ttnn.deallocate(out)

            # B: `from_torch` builds the tile padding as zeros, so this arm differs from A
            # only in what lies past the logical end of the tensor.
            host = ttnn.to_torch(a1)
            a1b = ttnn.from_torch(host, dtype=a1.dtype, layout=ttnn.TILE_LAYOUT, device=device)
            out, _ = conv(a1b, frames, 1)
            cells.append(summarise(out))
            ttnn.deallocate(out)
            ttnn.deallocate(a1b)

            # C: the same zeroing without leaving the device -- row-major has no padding
            # rows to carry, so re-tiling has to synthesise them.
            try:
                rm = ttnn.to_layout(a1, ttnn.ROW_MAJOR_LAYOUT)
                a1c = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
                ttnn.deallocate(rm)
                out, _ = conv(a1c, frames, 1)
                cells.append(summarise(out))
                ttnn.deallocate(out)
                ttnn.deallocate(a1c)
            except Exception as exc:  # noqa: BLE001
                cells.append((-1, float("nan")))
                print(f"      (device re-tile failed: {str(exc)[:70]})")

            pad = (-frames) % 32
            row = f"  {frames:>8}{pad:>10}{in_max:>12.3f}"
            for n_inf, mx in cells:
                row += f"{f'{n_inf} inf, max {mx:.3g}':>22}"
            print(row)

            for t in (a1, si, x):
                ttnn.deallocate(t)

        print("\n  If B and C are clean where A is not, the convolution is reading rows past the")
        print("  logical end of its input, and re-tiling before it is a workable model-side fix.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
