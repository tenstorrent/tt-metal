# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""What makes the model's conv1d fail where a bare one does not?

`repro_conv1d_wormhole.py` calls `ttnn.conv1d` at the failing geometry -- `Conv1d(128 ->
128, k=11, pad=5)` over `[1, 8321, 128]`, HiFi4 + fp32 accumulation, on Wormhole -- with a
random weight and a random input, and gets the right answer at **every** length. The model's
own call at the identical geometry returns `1.58e38`.

So the length is necessary but not sufficient, and something about *this conv object*
completes it. Four candidates, and one arm each:

    A  model conv (prepared weights) + model input      the failing path
    B  model conv                    + random input     is it the data?
    C  bare ttnn.conv1d, model weight + model input     is it `prepare_conv_weights`?
    D  bare ttnn.conv1d, model weight + random input    weight alone
    E  bare, random weight           + model input      input alone

`TtConv1d` hoists weight preparation out of the op with `ttnn.prepare_conv_weights` so that
convolutions can be captured in a trace (the op otherwise transfers weights at call time,
which a trace rejects). That is the one structural difference between the model's call and
the bare one, which makes C the arm to read first: **if C is clean where A is not, the
prepared-weight layout is wrong at this geometry** and the bare-call repro was never going
to show it.

    python3 models/demos/cosyvoice/scripts/probe_conv_bisect.py [--frames 8321]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GOLDEN = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")


def worst(t) -> tuple[float, int]:
    v = ttnn.to_torch(t).float()
    fin = v[torch.isfinite(v)]
    return (float(fin.abs().max()) if fin.numel() else float("nan"), int((~torch.isfinite(v)).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8321)
    ap.add_argument("--l1", type=int, default=131072)
    args = ap.parse_args()
    frames = args.frames

    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.weights import WeightBag

    device = ttnn.open_device(device_id=0, l1_small_size=args.l1)
    try:
        hift = TtHiFTGenerator(device, WeightBag.load(os.path.join(GOLDEN, "hift_weights.npz")))
        down, rb = hift.source_downs[1], hift.source_resblocks[1]
        conv = rb.convs1[0]

        # The model's real input to the failing conv.
        torch.manual_seed(1986)
        seed = ttnn.from_torch(
            torch.randn(1, frames, 18) * 0.06, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        si, _ = down(seed, frames, 1)
        real_in = rb.act1[0](si)
        real_t = ttnn.to_torch(real_in).float()

        torch.manual_seed(7)
        rand_in = ttnn.from_torch(
            torch.randn(1, frames, conv.in_channels) * float(real_t.abs().max()) / 3.0,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        w_model = conv._weight_4d  # OIHW, H=1, as ttnn.conv1d wants
        w_rand = ttnn.from_torch(
            torch.randn(conv.out_channels, conv.in_channels, 1, conv.kernel_size) * 0.05, dtype=ttnn.bfloat16
        )

        def bare(w, x, bias):
            out, _ = ttnn.conv1d(
                input_tensor=x,
                weight_tensor=w,
                bias_tensor=bias,
                device=device,
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                batch_size=1,
                input_length=frames,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                conv_config=conv.conv_config,
                compute_config=conv.compute_config,
                dtype=conv.dtype,
                return_output_dim=True,
            )
            return out

        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}   frames {frames}")
        print(
            f"  Conv1d({conv.in_channels} -> {conv.out_channels}, k={conv.kernel_size}, "
            f"pad={conv.padding}, dil={conv.dilation}), bias={conv.bias is not None}"
        )
        print(f"  model input max {float(real_t.abs().max()):.4f}\n")
        print(f"  {'arm':<44}{'max|out|':>14}{'n_inf':>9}{'  verdict'}")
        print("  " + "-" * 76)

        arms = [
            ("A  model conv (prepared w) + model input", lambda: conv(real_in, frames, 1)[0]),
            ("B  model conv (prepared w) + random input", lambda: conv(rand_in, frames, 1)[0]),
            ("C  bare conv1d, model weight + model input", lambda: bare(w_model, real_in, conv.bias)),
            ("D  bare conv1d, model weight + random input", lambda: bare(w_model, rand_in, conv.bias)),
            ("E  bare conv1d, random weight + model input", lambda: bare(w_rand, real_in, None)),
        ]
        for label, fn in arms:
            try:
                out = fn()
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:<44}{'FAILED ' + str(exc)[:40]}")
                continue
            mx, n = worst(out)
            print(f"  {label:<44}{mx:>14.4g}{n:>9}{'   <-- WRONG' if (n or mx > 1e4) else ''}")
            ttnn.deallocate(out)

        print("\n  C clean + A wrong  => prepare_conv_weights is the variable.")
        print("  C wrong + E clean  => the weight values matter, not the code path.")
        print("  D wrong            => the input is irrelevant; weight + geometry is enough.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
