# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Is the Wormhole conv1d corruption in the prepared weight, and does skipping preparation fix it?

At the failing geometry the model's convolution returns the same wrong value for a
completely different input, while a bare `ttnn.conv1d` with the same weight is correct
(`probe_conv_bisect.py`). An output that no longer depends on the activation points at
the operand that did not change: the weight `ttnn.prepare_conv_weights` produced.

This establishes the two things an upstream report needs:

  1. the prepared weight's own magnitude, at a length that works and one that does not;
  2. that the unprepared weight is a working fix, and what it costs -- the fallback
     `TtConv1d._prepared` takes when preparation raises, taken deliberately.

It is also the check to rerun once the upstream defect is fixed (`docs/VALIDATION.md`).

    python3 models/experimental/cosyvoice/scripts/probe_prepared_weights.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GOLDEN = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")

# 8129 works, 8193/8321/8577 do not, 8705 works.
DEFAULT = "8129,8193,8321,8577,8705"


def stats(t):
    v = ttnn.to_torch(t).float()
    fin = v[torch.isfinite(v)]
    return (float(fin.abs().max()) if fin.numel() else float("nan"), int((~torch.isfinite(v)).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=DEFAULT)
    ap.add_argument("--l1", type=int, default=131072)
    args = ap.parse_args()

    from models.experimental.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.experimental.cosyvoice.tt.weights import WeightBag

    device = ttnn.open_device(device_id=0, l1_small_size=args.l1)
    try:
        hift = TtHiFTGenerator(device, WeightBag.load(os.path.join(GOLDEN, "hift_weights.npz")))
        rb = hift.source_resblocks[1]
        conv = rb.convs1[0]
        host_max = float(ttnn.to_torch(conv._weight_4d).float().abs().max())

        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}")
        print(f"  Conv1d({conv.in_channels} -> {conv.out_channels}, k={conv.kernel_size}, pad={conv.padding})")
        print(f"  weight as stored: max|w| = {host_max:.4f}\n")
        print(
            f"  {'frames':>8}{'prepared max|w|':>18}{'n_inf':>8}{'out, prepared':>16}{'out, raw weight':>18}{'  ms prep / raw'}"
        )
        print("  " + "-" * 96)

        for frames in [int(x) for x in args.frames.split(",")]:
            torch.manual_seed(3)
            x = ttnn.from_torch(
                torch.randn(1, frames, conv.in_channels) * 0.5,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            w_prep, b_prep = conv._prepared(x, frames, 1)
            wmax, wbad = stats(w_prep)

            def run(w, b):
                out, _ = ttnn.conv1d(
                    input_tensor=x,
                    weight_tensor=w,
                    bias_tensor=b,
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

            timings = []
            outs = []
            for w, b in ((w_prep, b_prep), (conv.weight, conv.bias)):
                ttnn.deallocate(run(w, b))  # warm
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                o = run(w, b)
                ttnn.synchronize_device(device)
                timings.append((time.perf_counter() - t0) * 1e3)
                outs.append(stats(o))
                ttnn.deallocate(o)

            flag = "  <-- WRONG" if outs[0][0] > 1e4 else ""
            print(
                f"  {frames:>8}{wmax:>18.4g}{wbad:>8}{outs[0][0]:>16.4g}{outs[1][0]:>18.4g}"
                f"   {timings[0]:.2f} / {timings[1]:.2f}{flag}"
            )
            ttnn.deallocate(x)

        print("\n  A prepared weight far above the stored one is the defect, stated directly.")
        print("  The 'raw weight' column is the model-side fix; the timings are what it costs.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
