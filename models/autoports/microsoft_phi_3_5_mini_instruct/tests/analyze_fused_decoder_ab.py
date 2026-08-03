# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Summarize paired/interleaved fused-decoder benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def _inputs(paths):
    for path_text in paths:
        path = Path(path_text)
        if path.is_dir():
            yield from sorted(path.glob("*.json"))
        else:
            yield path


def _percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_ci(runs, statistic, *, resamples, seed):
    """Hierarchical bootstrap: resample processes, then pairs within each process."""
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sampled_runs = rng.choices(runs, k=len(runs))
        sampled_pairs = []
        for run in sampled_runs:
            sampled_pairs.extend(rng.choices(run, k=len(run)))
        estimates.append(statistic(sampled_pairs))
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Benchmark JSON files or directories")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    groups = defaultdict(list)
    for path in _inputs(args.paths):
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1 or data.get("mode") != "traced_decode":
            raise ValueError(f"{path}: unsupported benchmark JSON")
        key = (int(data["batch"]), ",".join(data["order"]))
        groups[key].append([float(sample["fused_minus_functional_ms"]) for sample in data["samples"]])

    if not groups:
        raise SystemExit("no benchmark JSON files found")

    print("delta_ms = fused - functional (negative favors fused)")
    for group_index, ((batch, order), runs) in enumerate(sorted(groups.items())):
        deltas = [delta for run in runs for delta in run]
        mean_ci = _bootstrap_ci(
            runs, statistics.mean, resamples=args.bootstrap_resamples, seed=args.seed + 2 * group_index
        )
        median_ci = _bootstrap_ci(
            runs, statistics.median, resamples=args.bootstrap_resamples, seed=args.seed + 2 * group_index + 1
        )
        print(
            f"batch={batch} order={order} files={len(runs)} pairs={len(deltas)} "
            f"paired_mean={statistics.mean(deltas):.6f} 95%CI=[{mean_ci[0]:.6f},{mean_ci[1]:.6f}] "
            f"paired_median={statistics.median(deltas):.6f} 95%CI=[{median_ci[0]:.6f},{median_ci[1]:.6f}]"
        )


if __name__ == "__main__":
    main()
