# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Search every contiguous PP layer split and score it by max stage cost.

Pipeline throughput is 1/max(stage), so the split that matters is the one with the smallest
SLOWEST stage -- not the one with the most even layer counts. For Gemma 4 those are different
things: `layer_types` is 50 sliding + 10 full with a global every 6th layer, and a global layer
costs several times a sliding one once the KV cache is deep.

The two per-layer costs default to values FITTED to a measured 15,15,15,15 run at 256k / chunk 8192
(cost summed over all 32 chunks). Fitted on that split, they predict the 17,13,17,13 stages to
under 1% -- so the model is worth trusting for enumeration, but re-fit it if the shape of the run
changes (different context, chunk size, or TP).

    python models/demos/gemma4/tests/perf/pp4/optimal_layer_split.py [--ranks 4] [--top 8]

Host-only; no device, no weights.
"""

import argparse
import itertools

# Measured on bh-glx-120-b03u02: per-layer cost summed over the 32 chunks of a 256k prefill at
# CP=8 x TP=1. A global layer pays the whole prefix; a sliding layer sees only its 1024-token
# window and is flat in context. The 7.4x ratio is the entire reason this script exists.
SLIDING_COST = 0.301
GLOBAL_COST = 2.234


def layer_types(num_layers: int, period: int = 6):
    """Gemma 4-31B: every `period`-th layer is full_attention (5, 11, ... 59)."""
    return ["full_attention" if (i % period) == period - 1 else "sliding_attention" for i in range(num_layers)]


def stage_cost(types, first, count, sliding, glob):
    g = sum(1 for i in range(first, first + count) if types[i] == "full_attention")
    return g * glob + (count - g) * sliding, g


def enumerate_splits(num_layers: int, ranks: int, types, sliding, glob):
    """Every way to cut `num_layers` into `ranks` contiguous non-empty stages."""
    for cuts in itertools.combinations(range(1, num_layers), ranks - 1):
        bounds = (0,) + cuts + (num_layers,)
        counts = [b - a for a, b in zip(bounds, bounds[1:])]
        costs, globals_ = [], []
        for first, n in zip(bounds, counts):
            c, g = stage_cost(types, first, n, sliding, glob)
            costs.append(c)
            globals_.append(g)
        yield max(costs), tuple(counts), tuple(globals_), tuple(round(c, 2) for c in costs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--sliding-cost", type=float, default=SLIDING_COST)
    ap.add_argument("--global-cost", type=float, default=GLOBAL_COST)
    args = ap.parse_args()

    sliding, glob = args.sliding_cost, args.global_cost
    types = layer_types(args.layers)
    n_global = types.count("full_attention")
    ideal = (types.count("sliding_attention") * sliding + n_global * glob) / args.ranks

    results = sorted(enumerate_splits(args.layers, args.ranks, types, sliding, glob))
    print(
        f"{args.layers} layers ({n_global} global), {args.ranks} ranks, "
        f"{len(results)} contiguous splits | sliding={sliding}s global={glob}s"
    )
    print(f"perfectly balanced (generally UNREACHABLE) = {ideal:.3f}s per stage\n")
    print(f"{'max stage':>10}  {'counts':>18}  {'globals':>14}  stage costs")
    for m, counts, globs, costs in results[: args.top]:
        print(f"{m:>9.3f}s  {str(counts):>18}  {str(globs):>14}  {list(costs)}")

    best = results[0]
    print(
        f"\nbest = {','.join(map(str, best[1]))}  ({best[0]:.3f}s, "
        f"{100 * (best[0] / ideal - 1):.1f}% above the ideal)"
    )
    if n_global % args.ranks:
        print(
            f"note: {n_global} global layers do not divide by {args.ranks}, so a perfectly balanced\n"
            f"      split does not exist at any layer count -- the gap above is STRUCTURAL."
        )
    print(f"\nuse it with:  PREFILL_PP_LAYER_COUNTS={','.join(map(str, best[1]))}")


if __name__ == "__main__":
    main()
