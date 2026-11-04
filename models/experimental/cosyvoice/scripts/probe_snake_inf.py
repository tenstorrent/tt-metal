# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Which of Snake's five ops returns `inf` on Wormhole, and what avoids it?

On Wormhole, `snake act2[0]` in the vocoder's source branch comes back non-finite for
`stft_frames` in [8193, 8577], where Blackhole runs the same lengths finite. Snake is
`x + sin^2(alpha*x)/alpha`, and each op stays finite for finite input of ordinary
magnitude: `sin` is in [-1, 1], its square in [0, 1], and `1/alpha` is a host-folded
constant.

`act1[0]` and `act2[0]` are the same activation on tensors of the same logical shape in
the same call, and only the second fails. What differs is the producer -- `act1[0]`
reads a `k=1` convolution's output and `act2[0]` a dilated one -- and convolutions
choose their output sharding from their own geometry. So this:

  1. prints the memory config of both activations' inputs;
  2. walks Snake one op at a time on the failing tensor, reporting magnitude as well as
     finiteness, so the report names a kernel -- or shows the input already at 1e38,
     which is finite, from the convolution before it;
  3. reruns the computation with the input moved to interleaved DRAM and to interleaved
     L1, either of which would also be a workaround.

On Wormhole the input is the fault: the convolution before Snake is the prepared-weight
defect in `docs/VALIDATION.md`.

    python3 models/experimental/cosyvoice/scripts/probe_snake_inf.py [--frames 8321]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GOLDEN = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")


def bad(t) -> int:
    return int((~torch.isfinite(ttnn.to_torch(t).float())).sum())


def describe(t) -> str:
    mc = t.memory_config()
    shard = getattr(mc, "shard_spec", None)
    s = f"{mc.memory_layout}".split(".")[-1] + "/" + f"{mc.buffer_type}".split(".")[-1]
    if shard is not None:
        s += f" shard{tuple(shard.shape)}"
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8321)  # L=130, the streaming case
    ap.add_argument("--l1", type=int, default=131072)
    args = ap.parse_args()
    frames = args.frames

    from models.experimental.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.experimental.cosyvoice.tt.weights import WeightBag

    device = ttnn.open_device(device_id=0, l1_small_size=args.l1)
    try:
        hift = TtHiFTGenerator(device, WeightBag.load(os.path.join(GOLDEN, "hift_weights.npz")))
        down, rb = hift.source_downs[1], hift.source_resblocks[1]
        snake = rb.act2[0]

        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}   frames {frames}")

        torch.manual_seed(1986)
        x = ttnn.from_torch(
            torch.randn(1, frames, 18) * 0.06, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        si, _ = down(x, frames, 1)
        a1 = rb.act1[0](si)
        c1, _ = rb.convs1[0](a1, frames, 1)

        # `n_inf` alone is not enough: a convolution can return values of 1e38 -- finite,
        # so every isfinite check passes -- and the activation downstream then overflows.
        # Magnitude is the diagnostic; finiteness is the symptom.
        print(f"\n  {'tensor':<30}{'shape':<18}{'memory config':<26}{'n_inf':>7}{'max|finite|':>14}")
        print("  " + "-" * 96)
        for name, t in (
            ("source_downs[1] out -> act1", si),
            ("act1[0] out -> convs1[0]", a1),
            ("convs1[0] out -> act2  (FAILS)", c1),
        ):
            v = ttnn.to_torch(t).float()
            fin = v[torch.isfinite(v)]
            mx = float(fin.abs().max()) if fin.numel() else float("nan")
            print(
                f"  {name:<30}{str(tuple(t.shape)):<18}{describe(t):<26}{int((~torch.isfinite(v)).sum()):>7}{mx:>14.4g}"
            )

        # --- Snake, one op at a time -----------------------------------------
        print(f"\n  Snake = x + sin^2(alpha*x)/alpha, on convs1[0]'s output")
        print(f"  {'step':<28}{'memory config':<26}{'n_inf':>9}{'max|finite|':>14}")
        print("  " + "-" * 79)

        def show(label, t):
            v = ttnn.to_torch(t).float()
            fin = v[torch.isfinite(v)]
            mx = float(fin.abs().max()) if fin.numel() else float("nan")
            print(f"  {label:<28}{describe(t):<26}{int((~torch.isfinite(v)).sum()):>9}{mx:>14.4g}")

        t1 = ttnn.multiply(c1, snake.alpha)
        show("multiply(x, alpha)", t1)
        t2 = ttnn.sin(t1)
        show("sin(.)", t2)
        t3 = ttnn.square(t2)
        show("square(.)", t3)
        t4 = ttnn.multiply(t3, snake.inv_alpha)
        show("multiply(., 1/alpha)", t4)
        t5 = ttnn.add(c1, t4)
        show("add(x, .)", t5)
        for t in (t1, t2, t3, t4, t5):
            ttnn.deallocate(t)

        # --- does a different memory config fix it? ---------------------------
        print(f"\n  the same Snake, with the input moved first")
        print(f"  {'placement':<28}{'memory config':<26}{'n_inf':>9}{'max|finite|':>14}")
        print("  " + "-" * 79)
        for label, mc in (("interleaved DRAM", ttnn.DRAM_MEMORY_CONFIG),):
            try:
                moved = ttnn.to_memory_config(c1, mc)
                out = snake(moved)
                show(label, out)
                ttnn.deallocate(out)
                ttnn.deallocate(moved)
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:<28}FAILED {str(exc)[:60]}")

        # A host-side reference, so "the right answer" is on the same page as the wrong one.
        ref = ttnn.to_torch(c1).float()
        alpha = ttnn.to_torch(snake.alpha).float().reshape(1, 1, -1)
        want = ref + (1.0 / (alpha + 1e-9)) * torch.sin(alpha * ref).pow(2)
        print(
            f"\n  torch on the same input: n_inf {int((~torch.isfinite(want)).sum())}, max|.| {float(want.abs().max()):.4f}"
        )
        print("  A bounded function of a finite input returning inf is a kernel defect, not a model one.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
