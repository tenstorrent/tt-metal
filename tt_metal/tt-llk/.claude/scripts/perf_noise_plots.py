# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Plot the run-to-run noise of one perf configuration, from its points CSV.

Reads the ``noise_report.points.csv`` that ``perf_noise_analysis.py`` writes and
produces four figures. The per-point CSV is far too large to keep in the repo;
these PNGs are what gets committed and reviewed.

    python perf_noise_plots.py <points.csv> "<title>" <out-dir> <prefix>

**rule** -- absolute movement against point size, log-log, one colour per marker.
The two clauses of the gate rule are drawn as lines: a horizontal floor at 30
cycles and a diagonal at 2%, with the wedge above both shaded. This is the
figure that answers whether small-marker jitter is a fixed cycle cost (a flat
band, so the floor is the right instrument) or proportional (a rising band, so a
percentage is).

**by_test** -- worst movement per test, so "which tests are unstable" needs no
pandas to answer.

**by_run_type** -- movement split by run type and marker; only drawn when the
configuration has more than one run type.

**bimodal** -- the five run values of the worst offending points, each normalised
to its own minimum. Jitter scatters; a discrete state change puts the runs into
two flat clusters.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # no display on a lab machine
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REL_THRESHOLD = 0.02
ABS_THRESHOLD = 30.0

# Scatter of 300k points is slow to draw and heavy as a PNG; the shape is
# identical at 60k. Sampling is deterministic so re-running gives the same figure.
_MAX_SCATTER = 60_000
_MARKER_COLOURS = {
    "TILE_LOOP": "#1f77b4",
    "KERNEL": "#2ca02c",
    "INIT": "#d62728",
    "UNINIT": "#ff7f0e",
}


def _load(path):
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame[(frame["median"] > 0) & (frame["abs_spread"] >= 0)]


def _fires(frame):
    return frame[
        (frame["spread"] > REL_THRESHOLD) & (frame["abs_spread"] > ABS_THRESHOLD)
    ]


def plot_rule(frame, title, path):
    """Absolute movement vs point size, with both clauses of the rule drawn."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    shown = frame
    if len(shown) > _MAX_SCATTER:
        shown = shown.sample(_MAX_SCATTER, random_state=0)

    for marker, group in shown.groupby("marker"):
        ax.scatter(
            group["median"],
            group["abs_spread"].clip(lower=0.5),  # log axis cannot show exact zeros
            s=4,
            alpha=0.25,
            linewidths=0,
            label=f"{marker} (n={len(frame[frame['marker'] == marker]):,})",
            color=_MARKER_COLOURS.get(marker),
            rasterized=True,
        )

    lo = max(frame["median"].min(), 1)
    hi = frame["median"].max()

    # Shade the wedge the rule actually fires in: above the floor AND above 2%.
    # Drawn before the lines so the boundaries stay crisp.
    edge = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    ax.fill_between(
        edge,
        [max(ABS_THRESHOLD, x * REL_THRESHOLD) for x in edge],
        frame["abs_spread"].max() * 2,
        color="#d62728",
        alpha=0.07,
        zorder=0,
        label="rule fires here",
    )

    ax.axhline(
        ABS_THRESHOLD,
        color="black",
        lw=1.4,
        ls="--",
        label=f"{ABS_THRESHOLD:g} cycles",
    )
    ax.plot(
        [lo, hi],
        [lo * REL_THRESHOLD, hi * REL_THRESHOLD],
        color="black",
        lw=1.4,
        ls=":",
        label=f"{REL_THRESHOLD:.0%}",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("point size — median cycles across the 5 runs")
    ax.set_ylabel("movement — max minus min, cycles")
    ax.set_title(
        f"{title}\nthe rule fires only above BOTH lines "
        f"({len(_fires(frame)):,} of {len(frame):,} points)"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_by_test(frame, title, path):
    """Worst movement per test, sorted, flagged where the rule fires."""
    worst = frame.groupby("test")["spread"].max().sort_values()
    fired = set(_fires(frame)["test"].unique())
    colours = ["#d62728" if t in fired else "#1f77b4" for t in worst.index]

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.28 * len(worst) + 1.5)))
    ax.barh(range(len(worst)), worst.values * 100, color=colours)
    ax.set_yticks(range(len(worst)))
    ax.set_yticklabels(worst.index, fontsize=8)
    ax.axvline(REL_THRESHOLD * 100, color="black", lw=1.2, ls=":")
    ax.set_xlabel("worst movement of any point in the test (%)")
    ax.set_title(f"{title}\nred = the rule fires somewhere in this test")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_by_run_type(frame, title, path):
    """Movement split by run type and marker. Skipped for single-run-type sets.

    Percentile bars rather than a box plot: most points move exactly 0%, so every
    quartile collapses to zero and a box carries no information. The upper tail
    is the whole story, so plot it directly.
    """
    if frame["run_type"].nunique() < 2:
        return False

    quantiles = [(0.95, "p95"), (0.99, "p99"), (1.0, "max")]
    rows = []
    for (run_type, marker), group in frame.groupby(["run_type", "marker"]):
        rows.append(
            (
                f"{run_type}\n{marker}",
                [group["spread"].quantile(q) * 100 for q, _ in quantiles],
            )
        )

    positions = range(len(rows))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(9, 1.15 * len(rows)), 6))
    for index, (_, name) in enumerate(quantiles):
        ax.bar(
            [p + (index - 1) * width for p in positions],
            [values[index] for _, values in rows],
            width=width,
            label=name,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels([label for label, _ in rows], fontsize=7)
    ax.axhline(REL_THRESHOLD * 100, color="black", lw=1.2, ls=":", label="2%")
    ax.set_ylabel("movement (%)")
    ax.set_title(f"{title}\nupper tail of the movement distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_bimodal(frame, title, path, limit=14):
    """The 5 run values of the worst offenders, each normalised to its own median.

    Normalising to the minimum, as this did first, forces every series to read as
    "four low, one high" no matter which way the odd run actually went, and that
    misreading survived into a written conclusion. The median keeps the sign.
    """
    runs = [c for c in frame.columns if c.startswith("run_") and c[4:].isdigit()]
    worst = _fires(frame).nlargest(limit, "spread")
    if worst.empty or not runs:
        return False

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in worst.iterrows():
        values = row[runs].astype(float)
        base = values.median()
        ax.plot(
            range(1, len(runs) + 1),
            (values / base - 1) * 100,
            marker="o",
            ms=5,
            lw=0.8,
            alpha=0.85,
            label=f"{row['test']} {row['marker']} ({base:.0f} cy)",
        )

    ax.set_xticks(range(1, len(runs) + 1))
    ax.set_xlabel("run")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("against that point's own median (%)")
    ax.set_title(
        f"{title}\nworst {len(worst)} offenders — two levels mean a discrete "
        "state change, scatter means jitter; sign shows direction"
    )
    ax.legend(fontsize=6.5, loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        raise SystemExit(2)
    csv_path, title, out_dir, prefix = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    frame = _load(csv_path)

    plot_rule(frame, title, f"{out_dir}/{prefix}_rule.png")
    plot_by_test(frame, title, f"{out_dir}/{prefix}_by_test.png")
    written = [f"{prefix}_rule.png", f"{prefix}_by_test.png"]
    if plot_by_run_type(frame, title, f"{out_dir}/{prefix}_by_run_type.png"):
        written.append(f"{prefix}_by_run_type.png")
    if plot_bimodal(frame, title, f"{out_dir}/{prefix}_bimodal.png"):
        written.append(f"{prefix}_bimodal.png")

    print(f"{len(frame):,} points, {len(_fires(frame)):,} fire the rule")
    for name in written:
        print(f"  {out_dir}/{name}")


if __name__ == "__main__":
    main()
