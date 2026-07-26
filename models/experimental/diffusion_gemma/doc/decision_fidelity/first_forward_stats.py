#!/usr/bin/env python3
"""Decision statistics for a first-forward dump, and the diff between two of them.

Written after measuring the wrong thing. "TT 5.10 nats vs reference 3.75" compares MEAN per-position
entropy, but the sampler does not read the mean: EntropyBoundSampler accepts the k lowest-entropy
positions whose EXCLUSIVE prefix sum of entropies is <= entropy_bound (0.1 nats, absolute). What
decides k is therefore the LOW TAIL, and the reference's step-1 distribution is strongly bimodal --
on q106 the median position sits at 4.26 nats while four positions are essentially zero, which is why
a mean of 3.75 still yields only 6 accepted positions.

So the number to compare is the accept count and the tail that produces it, not the mean. TT's
collapse signature is accept pinned at 1, which is what "no position below the bound" looks like:
k=1 is always accepted because its exclusive prefix sum is 0.

Usage:
  first_forward_stats.py DUMP.pt [--json OUT.json]
  first_forward_stats.py REF.pt TT.pt          # side-by-side, plus per-layer hidden RMS
"""
import argparse
import json
import sys

import torch

ENTROPY_BOUND = 0.1


def accept_count(entropy: torch.Tensor, bound: float = ENTROPY_BOUND) -> int:
    """k = how many of the lowest-entropy positions the sampler accepts this step."""
    ordered = entropy.flatten().float().sort().values
    exclusive_prefix = torch.cat([torch.zeros(1), ordered.cumsum(0)[:-1]])
    return int((exclusive_prefix <= bound).sum())


def describe(dump: dict) -> dict:
    entropy = dump["entropy"].flatten().float()
    ordered = entropy.sort().values
    stats = dict(dump.get("logits_stats") or {})
    stats.update(
        {
            "entropy_mean": float(entropy.mean()),
            "entropy_median": float(ordered[len(ordered) // 2]),
            "entropy_min": float(ordered[0]),
            "entropy_max": float(ordered[-1]),
            "positions_below_0.01": int((entropy < 0.01).sum()),
            "positions_below_0.1": int((entropy < 0.1).sum()),
            "positions_below_0.5": int((entropy < 0.5).sum()),
            "positions_below_1.0": int((entropy < 1.0).sum()),
            "accept_count_step1": accept_count(entropy),
            "positions": int(entropy.numel()),
        }
    )
    return stats


def layer_rms(dump: dict) -> dict:
    out = {}
    for layer_idx, hidden in sorted((dump.get("layer_hidden") or {}).items()):
        h = hidden.float()
        out[int(layer_idx)] = {
            "rms": float(h.pow(2).mean().sqrt()),
            "mean_abs": float(h.abs().mean()),
            "max_abs": float(h.abs().max()),
        }
    return out


ORDER = [
    "accept_count_step1",
    "positions_below_0.1",
    "positions_below_0.01",
    "positions_below_0.5",
    "positions_below_1.0",
    "entropy_mean",
    "entropy_median",
    "entropy_min",
    "entropy_max",
    "logit_std",
    "logit_max_mean",
    "logit_min_mean",
    "top1_minus_top2_mean",
    "distinct_argmax",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", nargs="+", help="one dump, or REF.pt TT.pt to diff")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    loaded = [(path, torch.load(path, map_location="cpu", weights_only=False)) for path in args.dumps]
    described = [(path, describe(dump)) for path, dump in loaded]

    if len(described) == 1:
        path, stats = described[0]
        print(path)
        for key in ORDER:
            if key in stats:
                print(f"  {key:>22} = {stats[key]}")
    else:
        (ref_path, ref), (tt_path, tt) = described[0], described[1]
        print(f"{'':>22} {'reference':>14} {'TT':>14}   {'':>10}")
        print(f"{'':>22} {ref_path.split('/')[-1]:>14} {tt_path.split('/')[-1]:>14}")
        for key in ORDER:
            if key not in ref or key not in tt:
                continue
            a, b = ref[key], tt[key]
            if isinstance(a, float):
                ratio = f"x{b / a:.3f}" if a else ""
                print(f"  {key:>22} {a:>14.5f} {b:>14.5f}   {ratio:>10}")
            else:
                print(f"  {key:>22} {a:>14} {b:>14}")

        ref_layers, tt_layers = layer_rms(loaded[0][1]), layer_rms(loaded[1][1])
        shared = sorted(set(ref_layers) & set(tt_layers))
        if shared:
            print(f"\nper-layer hidden RMS  (ratio TT/ref -- the first layer to drift localises it)")
            print(f"{'layer':>6} {'ref rms':>10} {'TT rms':>10} {'ratio':>8} {'ref max|h|':>11} {'TT max|h|':>10}")
            for layer_idx in shared:
                r, t = ref_layers[layer_idx], tt_layers[layer_idx]
                ratio = t["rms"] / r["rms"] if r["rms"] else float("nan")
                print(
                    f"{layer_idx:>6} {r['rms']:>10.4f} {t['rms']:>10.4f} {ratio:>8.3f} "
                    f"{r['max_abs']:>11.2f} {t['max_abs']:>10.2f}"
                )

    if args.json:
        json.dump(
            {path: stats for path, stats in described} | {f"layers::{path}": layer_rms(dump) for path, dump in loaded},
            open(args.json, "w"),
            indent=2,
        )
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
