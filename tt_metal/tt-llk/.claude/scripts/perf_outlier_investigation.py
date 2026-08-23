# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Characterise the points a noise baseline flags, without assuming a cause.

Given one ``noise_report.points.csv``, this asks four questions about the points
that break the gate rule. Each is answerable from the data alone, and each can
falsify a claim rather than merely illustrate one.

1. **Shape.** How many distinct values do the five runs take? Two means a
   discrete state change. Five means continuous jitter.
2. **Size of the deviation.** For each flagged point, the odd run's value over
   the median of the other four. A tight cluster means a constant penalty; a
   broad spread means it is not one thing.
3. **Is the deviation per-run or per-point?** If one run is slow, are the same
   run's other points slow too? A run-level state predicts the odd run is shared
   across points; independent per-point events predict it is not. The report
   compares the observed concentration against what independence would give.
4. **Does any sweep parameter separate flagged from unflagged points?** Within
   the dominant test, each ``cfg_`` column is scored by how much the flag rate
   varies across its values. This is association only -- it cannot establish a
   cause, and the report says so.
5. **What threshold would cover everything?** For each absolute-cycle floor, the
   smallest percentage that flags nothing. This is exact rather than a percentile
   estimate, and it prices the alternative to fixing the outliers: adopting a
   threshold large enough to absorb them.

    python perf_outlier_investigation.py <points.csv> "<title>" <out.md> [RUN_TYPE]
"""

import sys

import pandas as pd

REL_THRESHOLD = 0.02
ABS_THRESHOLD = 30.0


def _run_columns(frame):
    return [c for c in frame.columns if c.startswith("run_") and c[4:].isdigit()]


def _load(path, run_type):
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    if run_type and "run_type" in frame.columns:
        frame = frame[frame["run_type"] == run_type]
    return frame


def _flagged(frame):
    return frame[
        (frame["spread"] > REL_THRESHOLD) & (frame["abs_spread"] > ABS_THRESHOLD)
    ]


def shape_of_runs(frame, runs):
    """How many distinct values the five runs take, as a distribution."""
    counts = frame[runs].round(0).nunique(axis=1).value_counts().sort_index()
    total = int(counts.sum())
    lines = ["| distinct values across the 5 runs | points | share |", "|---|--:|--:|"]
    for distinct, n in counts.items():
        lines.append(f"| {distinct} | {n:,} | {n / total:.1%} |")
    return lines


def deviation_size(frame, runs):
    """Odd run over the median of the other four, for each flagged point."""
    values = frame[runs].astype(float)
    median = values.median(axis=1)
    odd_column = (values.sub(median, axis=0)).abs().idxmax(axis=1)
    odd_value = values.to_numpy()[
        range(len(values)), [runs.index(c) for c in odd_column]
    ]
    rest_median = pd.Series(
        [values.iloc[i].drop(odd_column.iloc[i]).median() for i in range(len(values))],
        index=values.index,
    )
    ratio = (odd_value / rest_median - 1) * 100
    described = ratio.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    lines = ["| statistic | odd run vs the other four (%) |", "|---|--:|"]
    for key in ("min", "5%", "25%", "50%", "75%", "95%", "max"):
        lines.append(f"| {key} | {described[key]:+.2f}% |")
    lines.append(f"| mean | {described['mean']:+.2f}% |")
    lines.append(f"| std | {described['std']:.2f} |")
    return lines, odd_column


def per_run_or_per_point(odd_column, runs):
    """Is the odd run shared across points, or independent per point?

    Under independence each run should be the odd one roughly 1/len(runs) of the
    time. Strong concentration on one run indicates a run-level state.
    """
    counts = odd_column.value_counts().reindex(runs, fill_value=0)
    total = int(counts.sum())
    expected = total / len(runs)
    lines = [
        f"Under independence each run would be the odd one about {expected:,.0f} "
        f"times ({1 / len(runs):.0%}).",
        "",
        "| run | times it was the odd one | share |",
        "|---|--:|--:|",
    ]
    for run, n in counts.items():
        lines.append(f"| {run} | {n:,} | {n / total:.1%} |")
    largest = counts.max() / total if total else 0
    lines += [
        "",
        f"Largest share on any single run: **{largest:.1%}**. A run-level state "
        f"predicts this near 100%; independent per-point events predict near "
        f"{1 / len(runs):.0%}.",
    ]
    return lines


def config_association(frame, flagged, top=6):
    """Which sweep parameters separate flagged from unflagged, in the top test."""
    if flagged.empty or "test" not in frame.columns:
        return ["No flagged points, so nothing to compare."]
    test = flagged["test"].value_counts().index[0]
    subset = frame[frame["test"] == test]
    is_flagged = subset.index.isin(flagged.index)
    lines = [
        f"Restricted to `{test}` -- {len(flagged[flagged['test'] == test]):,} of "
        f"{len(flagged):,} flagged points -- so a test's own sweep does not "
        f"masquerade as a parameter effect. {len(subset):,} points in that test.",
        "",
        "| parameter | flag rate by value | spread in rate |",
        "|---|---|--:|",
    ]
    scored = []
    for column in [c for c in subset.columns if c.startswith("cfg_")]:
        if subset[column].nunique(dropna=True) < 2:
            continue
        rate = pd.Series(is_flagged, index=subset.index).groupby(subset[column]).mean()
        if rate.empty:
            continue
        scored.append((rate.max() - rate.min(), column, rate))
    scored.sort(reverse=True, key=lambda item: item[0])
    for delta, column, rate in scored[:top]:
        shown = ", ".join(
            f"{value}={share:.0%}" for value, share in rate.sort_values().items()
        )
        lines.append(f"| `{column[4:]}` | {shown} | {delta:.0%} |")
    if not scored:
        lines.append("| -- | no parameter varies within this test | -- |")
    lines += [
        "",
        "**This is association, not cause.** Sweep parameters are correlated with "
        "each other, so a high spread here identifies where to look, and nothing "
        "more.",
    ]
    return lines


_FLOORS = (0, 10, 20, 30, 50, 100, 200, 500)


def threshold_coverage(frame):
    """Smallest percentage with zero false failures, per absolute-cycle floor.

    ``spread`` is (max-min)/median over the runs, so the smallest rule that flags
    nothing is just above the largest spread among points the floor still admits.
    Raising the floor only helps where the offending points are small; where they
    are large it changes nothing, and that is the useful signal.
    """
    lines = [
        "| cycle floor | points above it | smallest % that flags nothing |",
        "|--:|--:|--:|",
    ]
    for floor in _FLOORS:
        eligible = frame[frame["abs_spread"] > floor]
        need = "n/a" if eligible.empty else f"{eligible['spread'].max() * 100:.2f}%"
        lines.append(f"| {floor} | {len(eligible):,} | {need} |")
    return lines


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(2)
    csv_path, title, out_path = sys.argv[1:4]
    run_type = sys.argv[4] if len(sys.argv) > 4 else None

    frame = _load(csv_path, run_type)
    runs = _run_columns(frame)
    flagged = _flagged(frame)

    out = [
        f"# Outlier characterisation -- {title}",
        "",
        f"Rule: more than {REL_THRESHOLD:.0%} AND more than {ABS_THRESHOLD:g} cycles.",
        f"Scope: {run_type or 'all run types'} -- **{len(flagged):,}** flagged of "
        f"**{len(frame):,}** points.",
        "",
    ]

    if flagged.empty or not runs:
        out.append("Nothing flagged; the remaining questions do not apply.")
        with open(out_path, "w") as handle:
            handle.write("\n".join(out) + "\n")
        print("\n".join(out))
        return

    out += ["## 1. Shape -- how many distinct values do the five runs take?", ""]
    out += shape_of_runs(flagged, runs)

    out += ["", "## 2. Size of the deviation", ""]
    size_lines, odd_column = deviation_size(flagged, runs)
    out += size_lines

    out += ["", "## 3. Is the deviation per-run or per-point?", ""]
    out += per_run_or_per_point(odd_column, runs)

    out += ["", "## 4. Does any sweep parameter separate flagged from unflagged?", ""]
    out += config_association(frame, flagged)

    out += ["", "## 5. What threshold would cover everything?", ""]
    out += threshold_coverage(frame)

    text = "\n".join(out) + "\n"
    with open(out_path, "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
