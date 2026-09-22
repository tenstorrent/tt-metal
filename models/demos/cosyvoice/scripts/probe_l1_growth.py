# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Measure L1_SMALL growth across vocoder geometries on one open device.

`docs/VALIDATION.md` records that `CosyVoiceTTNN.synthesize_batch` wedges the board on
the second utterance, and attributes it to per-geometry state in the vocoder's
`conv_transpose2d`/halo path that `release_caches()` does not free. That is a
description, not a measurement. This probe supplies the measurement, and it does so
without driving the device into the hang: it synthesises one utterance at a time and
reports the allocator's L1_SMALL occupancy after each, so the growth can be seen as a
curve rather than as a crash.

Two modes, because the interesting variable is geometry rather than utterance count:

  --arm same      the same prompt N times   -> one geometry, repeated
  --arm differing N distinct prompts        -> N geometries

If the growth is per-geometry as recorded, `same` should flatten after the first pass
and `differing` should climb. If both climb, the state is per-*call* and the geometry
story is wrong. `--lengths` varies the token budget per call, which is what actually
separates the two: capping every call to one budget gives identical geometries and a
flat curve that says nothing.

    python3 probe_l1_growth.py --arm differing --n 4 --max-tokens 96
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

GOLDEN_DIR = os.environ.get("COSYVOICE_GOLDEN", "")


def l1_small_used(device):
    """Bytes currently allocated in the L1_SMALL bank, or None if unavailable."""
    import ttnn

    try:
        view = ttnn.get_memory_view(device, ttnn.BufferType.L1_SMALL)
        return int(view.total_bytes_allocated_per_bank)
    except Exception as e:  # pragma: no cover - introspection is best effort
        print(f"    (L1_SMALL view unavailable: {type(e).__name__}: {e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("same", "differing"), default="differing")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96)
    # Geometry is the variable, and capping every utterance to the same token budget
    # hides it -- a first pass did exactly that and produced four identical 42240-sample
    # runs with L1_SMALL flat, which says nothing. `--lengths` varies the token budget
    # per call instead, so one prompt yields several distinct mel geometries.
    ap.add_argument("--lengths", type=str, default="", help="comma-separated per-call max_tokens")
    ap.add_argument("--l1-small", type=int, default=131072, help="l1_small_size for open_device")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()] or None

    import ttnn
    from models.demos.cosyvoice.tt.common import GOLDEN_DIR as GD
    from models.demos.cosyvoice.tt.pipeline import CosyVoiceTTNN, PromptContext
    from models.demos.cosyvoice.tt.weights import WeightBag, default_weights_path

    hift = default_weights_path()
    flow, llm = hift.replace("hift_", "flow_"), hift.replace("hift_", "llm_")
    inputs_dir = os.environ.get("COSYVOICE_INPUTS", os.path.join(GD, "inputs"))
    cases = sorted(glob.glob(os.path.join(inputs_dir, "*.npz")))
    if not cases:
        print(f"no prompt .npz in {inputs_dir}")
        return 2
    n = len(lengths) if lengths else args.n
    picked = [cases[0]] * n if args.arm == "same" else (cases * n)[:n]
    budgets = lengths if lengths else [args.max_tokens] * n

    device = ttnn.open_device(device_id=0, l1_small_size=args.l1_small, trace_region_size=402653184)
    print(f"arm={args.arm}  l1_small={args.l1_small}  lengths={budgets}")
    print(f"  baseline L1_SMALL after open: {l1_small_used(device)}")
    model = CosyVoiceTTNN(device, WeightBag.load(llm), WeightBag.load(flow), WeightBag.load(hift))
    print(f"  after model load:             {l1_small_used(device)}")

    prev = None
    for i, (path, budget) in enumerate(zip(picked, budgets)):
        ctx, meta = PromptContext.from_npz(path)
        wav, tokens = model.synthesize(ctx, sampler="greedy", max_tokens=budget)
        n_samples = wav.shape[1] * wav.shape[2] if len(wav.shape) == 3 else wav.shape[-1]
        ttnn.deallocate(wav)
        used = l1_small_used(device)
        delta = "" if prev is None or used is None else f"   delta {used - prev:+d}"
        print(
            f"  [{i}] {meta['mode']}/{meta['lang']:<3} {len(tokens):>4} tokens "
            f"{n_samples:>7} samples   L1_SMALL {used}{delta}",
            flush=True,
        )
        prev = used

    print("  releasing caches...", flush=True)
    try:
        model.release_caches()
        print(f"  after release_caches:         {l1_small_used(device)}")
    except Exception as e:
        print(f"  release_caches raised {type(e).__name__}: {e}")
    ttnn.close_device(device)
    print("PROBE_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
