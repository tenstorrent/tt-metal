# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Measure LLK perf run-to-run noise and recommend a merge-gate threshold.

Input: N snapshots of ``perf_data`` produced by running the SAME commit N times
on the SAME machine. Because the code is identical, every difference between two
runs is noise. The largest noise we see is the floor a regression threshold must
clear -- a threshold below it fires on nothing but jitter.

Two questions are answered separately:

1. **Per-point stability** -- for each (test, marker, run_type, sweep-config) point,
   the spread across the N runs (cv, min/max). Identifies unstable configs.
2. **Gate false-positive floor** -- the distribution of |delta| a gate would have
   computed between two *disjoint* groups of runs of the same code, for
   ``median-of-k`` with k=1 and k=2. The p99/max of that distribution is the
   threshold below which the gate is guaranteed to produce false positives.

Both a relative threshold and an absolute-cycle floor are recommended: small
points (a few hundred cycles) swing a large *percentage* on a tiny *absolute*
change, and a relative-only gate is dominated by them.

    python perf_noise_analysis.py --run run_1 --run run_2 ... --report noise.md

Each ``--run`` is a directory laid out like ``perf_data`` (``<test>/<test>.csv``),
or a single CSV file.

Deliberately dependency-light (pandas + stdlib) so it runs on any branch, exactly
like its sibling ``perf_regression_compare.py``.
"""

import argparse
import glob
import itertools
import os
import statistics
from collections import defaultdict

import pandas as pd

# A gate compares median-of-k vs median-of-k. Simulating k requires 2*k disjoint
# runs, so 5 runs support k=1 and k=2.
_SIM_GROUP_SIZES = (1, 2)


def _is_metric(col):
    return col.startswith("mean(") or col.startswith("std(")


def _is_ignored(col):
    # Not part of a point's identity: timing stats and per-stage code size.
    return _is_metric(col) or col.startswith("TEXT_SIZE(")


def _run_csvs(run_path):
    """The combined per-test CSVs in one run snapshot (.post/.counters excluded)."""
    if os.path.isfile(run_path):
        return [run_path]
    return sorted(
        p
        for p in glob.glob(os.path.join(run_path, "**", "*.csv"), recursive=True)
        if not p.endswith((".post.csv", ".counters.csv"))
    )


def _test_name(csv_path):
    return os.path.basename(csv_path)[: -len(".csv")]


def load_run(run_path):
    """One run snapshot -> {(test, marker, config, run_type): value}.

    ``config`` is every non-metric, non-marker column, so the key is stable
    without knowing any test's sweep axes up front.
    """
    point_values = {}
    for csv_path in _run_csvs(run_path):
        frame = pd.read_csv(csv_path)
        test = _test_name(csv_path)
        config_cols = [c for c in frame.columns if not _is_ignored(c) and c != "marker"]
        mean_cols = [c for c in frame.columns if c.startswith("mean(")]
        for _, row in frame.iterrows():
            config = tuple(sorted((c, row[c]) for c in config_cols if pd.notna(row[c])))
            for col in mean_cols:
                value = row.get(col)
                if pd.notna(value):
                    run_type = col[len("mean(") : -1]
                    key = (test, row.get("marker"), config, run_type)
                    point_values[key] = float(value)
    return point_values


def _percentile(sorted_values, fraction):
    """Nearest-rank percentile. Conservative (rounds up) -- a threshold derived
    from an interpolated percentile can sit below an observed sample."""
    if not sorted_values:
        return None
    rank = max(1, min(len(sorted_values), int(-(-fraction * len(sorted_values) // 1))))
    return sorted_values[rank - 1]


def _stats(values):
    lo, hi = min(values), max(values)
    med = statistics.median(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "min": lo,
        "max": hi,
        "median": med,
        "mean": mean,
        "std": sd,
        "cv": sd / mean if mean else 0.0,
        "spread": (hi - lo) / med if med else 0.0,
        "abs_spread": hi - lo,
    }


def _simulate_gate_deltas(values, group_size):
    """|delta| a median-of-k gate would report between two disjoint run groups.

    Returns a list of (abs_relative_delta, abs_cycle_delta). Groups must be
    disjoint: reusing a run on both sides correlates the two sides and
    understates the noise a real gate sees.
    """
    indices = range(len(values))
    deltas = []
    for base_idx in itertools.combinations(indices, group_size):
        remaining = [i for i in indices if i not in base_idx]
        for cur_idx in itertools.combinations(remaining, group_size):
            base = statistics.median([values[i] for i in base_idx])
            cur = statistics.median([values[i] for i in cur_idx])
            if base:
                deltas.append((abs(cur - base) / base, abs(cur - base)))
    return deltas


def analyze(run_paths, *, min_cycles=0.0):
    """Per-point stability + simulated gate-delta distributions.

    ``min_cycles`` drops points whose median is below the floor from the
    threshold recommendation (they are still reported), which is how you see
    what an absolute floor buys.
    """
    runs = [load_run(p) for p in run_paths]
    all_keys = set().union(*runs) if runs else set()

    points, incomplete = [], []
    for key in sorted(all_keys, key=repr):
        values = [r[key] for r in runs if key in r]
        if len(values) < len(runs):
            incomplete.append(
                {
                    "test": key[0],
                    "marker": key[1],
                    "run_type": key[3],
                    "config": dict(key[2]),
                    "present_in": len(values),
                    "of_runs": len(runs),
                }
            )
        if len(values) < 2:
            continue
        points.append(
            {
                "test": key[0],
                "marker": key[1],
                "run_type": key[3],
                "config": dict(key[2]),
                "values": values,
                **_stats(values),
            }
        )

    # Simulated gate deltas, sliced the ways a threshold might be scoped.
    sims = {}
    for k in _SIM_GROUP_SIZES:
        if len(runs) < 2 * k:
            continue
        overall, by_run_type, by_marker, by_test = (
            [],
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        for point in points:
            if point["median"] < min_cycles:
                continue
            for rel, absolute in _simulate_gate_deltas(point["values"], k):
                overall.append(rel)
                by_run_type[point["run_type"]].append(rel)
                by_marker[point["marker"]].append(rel)
                by_test[point["test"]].append(rel)
            point.setdefault("gate_delta_max", {})[k] = max(
                (d[0] for d in _simulate_gate_deltas(point["values"], k)), default=0.0
            )
        sims[k] = {
            "overall": sorted(overall),
            "by_run_type": {t: sorted(v) for t, v in by_run_type.items()},
            "by_marker": {t: sorted(v) for t, v in by_marker.items()},
            "by_test": {t: sorted(v) for t, v in by_test.items()},
        }

    return {
        "n_runs": len(runs),
        "points": points,
        "incomplete": incomplete,
        "sims": sims,
    }


def _summarize(sorted_deltas):
    return {
        "n": len(sorted_deltas),
        "p50": _percentile(sorted_deltas, 0.50),
        "p90": _percentile(sorted_deltas, 0.90),
        "p95": _percentile(sorted_deltas, 0.95),
        "p99": _percentile(sorted_deltas, 0.99),
        "max": sorted_deltas[-1] if sorted_deltas else None,
    }


def _round_up_threshold(value):
    """Round a noise floor up to a defensible gate threshold (1% granularity)."""
    if value is None:
        return None
    return max(0.01, -(-value * 100 // 1) / 100)


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}%"


def render_report(result, *, min_cycles, run_paths, meta=None):
    n = result["n_runs"]
    lines = [
        "# LLK perf run-to-run noise",
        "",
        f"- runs: **{n}** (same commit, same machine)",
        f"- points compared: **{len(result['points'])}** "
        "(test x marker x run_type x sweep-config)",
        f"- absolute floor applied to the recommendation: **{min_cycles:g} cycles**",
    ]
    for key, value in (meta or {}).items():
        lines.append(f"- {key}: {value}")
    lines += ["", "Runs analyzed:", ""]
    lines += [f"- `{p}`" for p in run_paths]

    recommendation = {}
    for k in sorted(result["sims"]):
        stats = _summarize(result["sims"][k]["overall"])
        recommendation[k] = stats
        lines += [
            "",
            f"## Gate false-positive floor -- median-of-{k} vs median-of-{k}",
            "",
            f"{stats['n']} simulated comparisons of identical code. Any threshold "
            "at or below a percentile below fires on that fraction of points "
            "**with no code change at all**.",
            "",
            "| percentile | |delta| |",
            "| --- | --- |",
            f"| p50 | {_pct(stats['p50'])} |",
            f"| p90 | {_pct(stats['p90'])} |",
            f"| p95 | {_pct(stats['p95'])} |",
            f"| p99 | {_pct(stats['p99'])} |",
            f"| max | {_pct(stats['max'])} |",
        ]

    lines += ["", "## Recommended threshold", ""]
    if recommendation:
        for k, stats in sorted(recommendation.items()):
            p99 = _round_up_threshold(stats["p99"])
            worst = _round_up_threshold(stats["max"])
            lines += [
                f"- **median-of-{k}**: `{_pct(p99)}` clears the p99 of pure noise; "
                f"`{_pct(worst)}` clears every observed noise sample.",
            ]
        lines += [
            "",
            f"Pair the relative threshold with an absolute floor of "
            f"**{min_cycles:g} cycles** -- flag a point only when it is both more "
            "than the relative threshold slower AND more than the floor slower in "
            "absolute cycles.",
        ]
    else:
        lines.append("- not enough runs to simulate a gate (need at least 2).")

    # Per-slice floors: one global threshold is set by the noisiest slice, so show
    # which slice that is before accepting the global number.
    for k in sorted(result["sims"]):
        for slice_name, label in (
            ("by_run_type", "run type"),
            ("by_marker", "marker"),
        ):
            rows = result["sims"][k][slice_name]
            if not rows:
                continue
            lines += [
                "",
                f"### Noise by {label} (median-of-{k})",
                "",
                f"| {label} | n | p95 | p99 | max |",
                "| --- | --- | --- | --- | --- |",
            ]
            ordered = sorted(
                rows.items(), key=lambda kv: _summarize(kv[1])["p99"] or 0, reverse=True
            )
            for name, deltas in ordered:
                s = _summarize(deltas)
                lines.append(
                    f"| {name} | {s['n']} | {_pct(s['p95'])} | {_pct(s['p99'])} "
                    f"| {_pct(s['max'])} |"
                )

    noisiest = sorted(result["points"], key=lambda p: p["spread"], reverse=True)[:25]
    if noisiest:
        lines += [
            "",
            "## 25 least stable points",
            "",
            "Candidates to stabilize, exclude from the gate, or give a per-point "
            "threshold. `spread` is (max-min)/median across the runs.",
            "",
            "| test | marker | run_type | median | spread | abs spread | config |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for p in noisiest:
            config = ", ".join(f"{k}={v}" for k, v in p["config"].items())
            lines.append(
                f"| {p['test']} | {p['marker']} | {p['run_type']} | "
                f"{p['median']:.1f} | {_pct(p['spread'])} | {p['abs_spread']:.1f} "
                f"| {config} |"
            )

    if result["incomplete"]:
        lines += [
            "",
            f"## Points missing from some runs ({len(result['incomplete'])})",
            "",
            "A point absent from a run means that config did not report in every "
            "run -- a flaky test or a partial sweep. The gate cannot compare these.",
            "",
            "| test | marker | run_type | present in | config |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in result["incomplete"][:25]:
            config = ", ".join(f"{k}={v}" for k, v in row["config"].items())
            lines.append(
                f"| {row['test']} | {row['marker']} | {row['run_type']} | "
                f"{row['present_in']}/{row['of_runs']} | {config} |"
            )

    return "\n".join(lines) + "\n"


def points_frame(result):
    rows = []
    for p in result["points"]:
        row = {
            "test": p["test"],
            "marker": p["marker"],
            "run_type": p["run_type"],
            **{f"cfg_{k}": v for k, v in p["config"].items()},
            "n": p["n"],
            "median": p["median"],
            "mean": p["mean"],
            "std": p["std"],
            "cv": p["cv"],
            "min": p["min"],
            "max": p["max"],
            "spread": p["spread"],
            "abs_spread": p["abs_spread"],
        }
        for k, value in (p.get("gate_delta_max") or {}).items():
            row[f"gate_delta_max_k{k}"] = value
        for i, value in enumerate(p["values"], start=1):
            row[f"run_{i}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="PATH",
        help="A run snapshot: a perf_data-shaped directory or a single CSV. "
        "Repeat once per run (5 recommended).",
    )
    parser.add_argument(
        "--min-cycles",
        type=float,
        default=100.0,
        help="Absolute floor: points with a median below this are excluded from "
        "the threshold recommendation (default: 100).",
    )
    parser.add_argument("--report", default="perf_noise_report.md")
    parser.add_argument(
        "--csv",
        default=None,
        help="Where to write per-point stats (default: <report>.points.csv).",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Provenance to record in the report, e.g. --label arch=wormhole.",
    )
    args = parser.parse_args()

    meta = dict(pair.split("=", 1) for pair in args.label if "=" in pair)
    result = analyze(args.run, min_cycles=args.min_cycles)
    report = render_report(
        result, min_cycles=args.min_cycles, run_paths=args.run, meta=meta
    )

    with open(args.report, "w") as handle:
        handle.write(report)
    csv_path = args.csv or f"{os.path.splitext(args.report)[0]}.points.csv"
    points_frame(result).to_csv(csv_path, index=False)

    print(report)
    print(f"per-point stats: {csv_path}")


if __name__ == "__main__":
    main()
