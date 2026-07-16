# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare paired fresh-process results from the disabled-overhead benchmark."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--equivalence-margin-percent", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def geometric_mean_ratio(log_ratios):
    return math.exp(statistics.fmean(log_ratios))


def main():
    args = parse_args()
    if len(args.baseline) != len(args.candidate):
        raise ValueError(
            "Baseline and candidate must contain the same number of paired process runs"
        )
    if len(args.baseline) < 4:
        raise ValueError("At least four paired process runs are required")

    baseline = [json.loads(path.read_text()) for path in args.baseline]
    candidate = [json.loads(path.read_text()) for path in args.candidate]
    baseline_commits = {run["git_commit"] for run in baseline}
    candidate_commits = {run["git_commit"] for run in candidate}
    if len(baseline_commits) != 1 or len(candidate_commits) != 1:
        raise ValueError("All runs in each group must come from one commit")
    if not all(run["execute_trace_is_direct_binding"] for run in baseline + candidate):
        raise ValueError("Every run must use the direct execute_trace binding")
    if any(run["tracker_environment"] for run in baseline + candidate):
        raise ValueError(
            "Every run must have all trace allocation tracker variables unset"
        )

    metric_names = baseline[0]["results"].keys()
    rng = random.Random(20260716)
    comparisons = {}

    for metric_name in metric_names:
        baseline_medians = [
            run["results"][metric_name]["median_ns_per_iteration"] for run in baseline
        ]
        candidate_medians = [
            run["results"][metric_name]["median_ns_per_iteration"] for run in candidate
        ]
        log_ratios = [
            math.log(candidate_ns / baseline_ns)
            for baseline_ns, candidate_ns in zip(baseline_medians, candidate_medians)
        ]

        bootstrap_ratios = []
        for _ in range(args.bootstrap_samples):
            resample = [rng.choice(log_ratios) for _ in log_ratios]
            bootstrap_ratios.append(geometric_mean_ratio(resample))
        bootstrap_ratios.sort()

        ratio = geometric_mean_ratio(log_ratios)
        lower_ratio = percentile(bootstrap_ratios, 0.025)
        upper_ratio = percentile(bootstrap_ratios, 0.975)
        upper_slowdown_percent = (upper_ratio - 1) * 100
        comparisons[metric_name] = {
            "baseline_median_ns_per_iteration": statistics.median(baseline_medians),
            "candidate_median_ns_per_iteration": statistics.median(candidate_medians),
            "paired_geometric_mean_change_percent": (ratio - 1) * 100,
            "change_95_percent_ci": [(lower_ratio - 1) * 100, upper_slowdown_percent],
            "equivalent_within_margin": upper_slowdown_percent
            <= args.equivalence_margin_percent,
        }

    payload = {
        "baseline_commit": baseline[0]["git_commit"],
        "candidate_commit": candidate[0]["git_commit"],
        "paired_process_runs": len(baseline),
        "equivalence_margin_percent": args.equivalence_margin_percent,
        "all_metrics_equivalent": all(
            result["equivalent_within_margin"] for result in comparisons.values()
        ),
        "comparisons": comparisons,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not payload["all_metrics_equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
