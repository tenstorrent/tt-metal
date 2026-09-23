# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Measure L1_SMALL occupancy across vocoder geometries on one open device.

Synthesises one utterance at a time and prints the allocator's L1_SMALL occupancy after
each, so growth shows as a curve rather than a crash. The mel geometry follows the
token count, so the per-call token budget is the variable:

  --arm same       one prompt for every call
  --arm differing  a different prompt for each call
  --lengths a,b,c  one call per budget; without it, `--n` calls at `--max-tokens`

Per-geometry growth climbs at each new budget and stays flat on a repeated one; per-call
growth climbs every time. Equal budgets give equal geometries and a flat curve, so give
`--lengths` distinct values. Seven geometries, on a bank large enough to hold them:

    python3 probe_l1_growth.py --arm same --l1-small 524288 --lengths 96,128,160,192,224,256,288
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
