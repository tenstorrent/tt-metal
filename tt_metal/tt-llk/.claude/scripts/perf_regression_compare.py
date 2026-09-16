# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained perf compare for the perf-regression-check skill.

Compares two perf runs of the same test — ``current`` against ``baseline``. The
two sides are usually two commits (a branch HEAD vs the commit it was branched
from, or any two commit hashes). Both sides may have several iterations; we take
the **median per point** on each side, flag points where current is more than
``threshold`` slower than baseline (regressions) and, symmetrically, points that
are more than ``threshold`` faster (improvements).

Deliberately dependency-light so it runs on any branch, merged or not: reads the
raw ``perf_data`` CSVs directly (no Parquet, no perf schema). A "point" is
``(marker, sweep-config)`` where the config is every column that is not a timing
or code-size column — so it needs nothing but the CSVs the sweep already writes.

    python perf_regression_compare.py \
        --current  'runs/current_*.csv' \
        --baseline 'runs/baseline_*.csv' \
        --threshold 0.05 --report regression_report.md
"""

import argparse
import glob
from statistics import median

import pandas as pd


def _is_metric(col):
    return col.startswith("mean(") or col.startswith("std(")


def _is_ignored(col):
    # Not part of a point's identity: timing stats and per-stage code size.
    return _is_metric(col) or col.startswith("TEXT_SIZE(")


def _point_key(row, config_cols):
    """A point's identity within one test: marker + the sweep configuration."""
    config = tuple(sorted((c, row[c]) for c in config_cols if pd.notna(row[c])))
    return (row.get("marker"), config)


def _medians(frames):
    """{(point_key, mean_col): median value} across a list of run DataFrames."""
    samples = {}
    for df in frames:
        config_cols = [c for c in df.columns if not _is_ignored(c) and c != "marker"]
        mean_cols = [c for c in df.columns if c.startswith("mean(")]
        for _, row in df.iterrows():
            key = _point_key(row, config_cols)
            for col in mean_cols:
                val = row.get(col)
                if pd.notna(val):
                    samples.setdefault((key, col), []).append(float(val))
    return {k: median(v) for k, v in samples.items()}


# Defaults are measured, not guessed: five runs of one commit on one card, over
# 108,377 points (L1_TO_L1) and 311,352 (isolates). The numbers below come from
# docs/perf_evaluation/results/blackhole-nonsol/README.md.
#
#   THRESHOLD   No TILE_LOOP or KERNEL measurement moved more than 1.9% between
#               identical runs. 2% sits above every observed sample and is still
#               tight enough to catch a real regression.
#   MIN_CYCLES  INIT and UNINIT are a few hundred cycles and wobble by up to 25.
#               That is a large *percentage* of a small number. 30 cycles clears
#               every wobble we saw.
#
# A point must exceed BOTH. Either clause alone is wrong: percentage-only fires on
# the small markers (474 of 108,377 points, all moving 25 cycles or less), and
# cycles-only fires on the big ones (TILE_LOOP moves up to 5,110 cycles and that
# is still only 1.9%). Together they flagged nothing across four of the five
# configurations measured.
DEFAULT_THRESHOLD = 0.02
DEFAULT_MIN_CYCLES = 30.0


def compare_runs(
    current_csvs,
    baseline_csvs,
    *,
    threshold=DEFAULT_THRESHOLD,
    min_cycles=DEFAULT_MIN_CYCLES,
):
    """Median-vs-median comparison.

    Returns ``{records, regressions, improvements, new_points, noise_filtered}``.
    ``delta`` is the fractional change vs baseline (0.12 = 12% slower, -0.12 = 12%
    faster) and ``abs_delta`` is the same change in cycles. A regression needs
    ``delta > threshold`` AND ``abs_delta > min_cycles``; an improvement is the
    mirror image. See the constants above for why both are required.

    ``noise_filtered`` counts points that cleared the percentage but not the cycle
    floor — exactly the points a relative-only rule would have failed on.
    """
    cur = _medians([pd.read_csv(p) for p in current_csvs])
    base = _medians([pd.read_csv(p) for p in baseline_csvs])

    records, regressions, improvements, new_points = [], [], [], []
    noise_filtered = 0
    for (key, mean_col), cval in cur.items():
        marker, config = key
        run_type = mean_col[len("mean(") : -1]
        point = {
            "marker": marker,
            "run_type": run_type,
            "config": dict(config),
            "current": cval,
        }
        bval = base.get((key, mean_col))
        if bval is None:
            new_points.append(point)  # no baseline (new config / new test)
            continue
        delta = (cval - bval) / bval if bval else 0.0
        abs_delta = cval - bval
        big_enough = abs(abs_delta) > min_cycles
        if abs(delta) > threshold and not big_enough:
            noise_filtered += 1
        record = {
            **point,
            "baseline": bval,
            "delta": delta,
            "abs_delta": abs_delta,
            "regression": delta > threshold and big_enough,
            "improvement": delta < -threshold and big_enough,
        }
        records.append(record)
        if record["regression"]:
            regressions.append(record)
        elif record["improvement"]:
            improvements.append(record)
    return {
        "records": records,
        "regressions": regressions,
        "improvements": improvements,
        "new_points": new_points,
        "noise_filtered": noise_filtered,
    }


_TOP_N = 25


def _side_line(role, sha, label, iters):
    """``- baseline (v1.2 tag): `abc123` — 3 iteration(s)``; label/iters optional."""
    named = f"{role} ({label})" if label else role
    tail = f" — {iters} iteration(s)" if iters else ""
    return f"- {named}: `{sha}`{tail}"


def _delta_table(rows, *, caption):
    """One Markdown table of points, worst delta first, config truncated to fit."""
    lines = [
        f"## {caption}",
        "",
        "| marker | run type | current | baseline | Δ | Δ cycles | config |",
        "|---|---|--:|--:|--:|--:|---|",
    ]
    for r in rows:
        cfg = ", ".join(f"{k}={v}" for k, v in sorted(r["config"].items()))
        if len(cfg) > 90:
            cfg = cfg[:87] + "…"
        lines.append(
            f"| {r['marker']} | {r['run_type']} | {r['current']:.1f} | "
            f"{r['baseline']:.1f} | {r['delta'] * 100:+.1f}% | "
            f"{r.get('abs_delta', 0.0):+.0f} | {cfg} |"
        )
    return lines


def render_report(
    result,
    *,
    threshold,
    test,
    baseline_sha,
    current_sha,
    min_cycles=DEFAULT_MIN_CYCLES,
    baseline_iters=None,
    current_iters=None,
    baseline_label=None,
    current_label=None,
):
    """A short Markdown report: verdict + worst regressions + improvements + new points.

    ``baseline_label``/``current_label`` say what each side *is* (``branch point on
    main``, a ref as the user typed it, …); the comparison itself is just
    current-vs-baseline, so any two commits work.

    Only the ``_TOP_N`` biggest regressions are tabled; every compared point goes
    to a companion ``*.points.csv`` and every regression to ``*.regressions.csv``
    (both written by main).
    """
    regs = sorted(result["regressions"], key=lambda r: -r["delta"])
    imps = sorted(result.get("improvements", []), key=lambda r: r["delta"])
    verdict = "❌ REGRESSIONS FOUND" if regs else "✅ no regressions"
    lines = [
        f"# Perf compare — {test}",
        "",
        f"**{verdict}**",
        "",
        f"Rule: a point is a regression when it is **more than {threshold * 100:.0f}% "
        f"slower AND more than {min_cycles:.0f} cycles slower**. Both must hold. "
        "Comparison is median-vs-median, per (marker, run type, sweep config).",
        "",
        _side_line("baseline", baseline_sha, baseline_label, baseline_iters),
        _side_line("current", current_sha, current_label, current_iters),
        f"- {len(result['records'])} points compared, "
        f"**{len(regs)} regression(s)**, {len(imps)} improvement(s), "
        f"{len(result['new_points'])} new point(s)",
    ]
    filtered = result.get("noise_filtered", 0)
    if filtered:
        lines.append(
            f"- {filtered} point(s) moved more than {threshold * 100:.0f}% but by "
            f"{min_cycles:.0f} cycles or fewer, so they are ignored. Small markers "
            "(INIT, UNINIT) are a few hundred cycles, where a handful of cycles of "
            "jitter looks like a large percentage."
        )
    lines.append("")
    if regs:
        shown = regs[:_TOP_N]
        lines += _delta_table(
            shown, caption=f"Top {len(shown)} regressions (slower on current)"
        )
        if len(regs) > _TOP_N:
            lines.append("")
            lines.append(
                f"_… and {len(regs) - _TOP_N} more — see the companion `.regressions.csv`._"
            )
        lines.append("")
    if imps:
        shown = imps[:_TOP_N]
        lines += _delta_table(
            shown, caption=f"Top {len(shown)} improvements (faster on current)"
        )
        if len(imps) > _TOP_N:
            lines.append("")
            lines.append(
                f"_… and {len(imps) - _TOP_N} more — see the companion `.points.csv`._"
            )
        lines.append("")
    if result["new_points"]:
        lines.append(
            f"## New points ({len(result['new_points'])}) — no baseline, not counted as regressions"
        )
        lines.append(
            "_These configs/markers exist at the current commit but not at the baseline commit._"
        )
    return "\n".join(lines)


def _points_csv(records):
    """Records as flat CSV rows (full config, nothing truncated), worst delta first."""
    rows = []
    for r in sorted(records, key=lambda x: -x["delta"]):
        row = {
            "marker": r["marker"],
            "run_type": r["run_type"],
            "current": r["current"],
            "baseline": r["baseline"],
            "delta_pct": round(r["delta"] * 100, 2),
            "delta_cycles": round(r.get("abs_delta", 0.0), 1),
        }
        row.update(r["config"])
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare current perf CSVs to a baseline's (usually two commits)."
    )
    ap.add_argument("--current", required=True, help="glob for current-run CSVs")
    ap.add_argument("--baseline", required=True, help="glob for baseline-run CSVs")
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"relative slowdown that counts as a regression (default {DEFAULT_THRESHOLD})",
    )
    ap.add_argument(
        "--min-cycles",
        type=float,
        default=DEFAULT_MIN_CYCLES,
        help="absolute slowdown, in cycles, that a point must ALSO exceed "
        f"(default {DEFAULT_MIN_CYCLES:.0f}). Stops small markers such as INIT "
        "from failing the gate on a few cycles of jitter. 0 disables the clause.",
    )
    ap.add_argument("--report", default="regression_report.md")
    ap.add_argument("--test", default="?")
    ap.add_argument("--baseline-sha", default="?")
    ap.add_argument("--current-sha", default="?")
    ap.add_argument(
        "--baseline-label",
        help="what the baseline side is, e.g. 'branch point on main' or the ref as typed",
    )
    ap.add_argument("--current-label", help="what the current side is")
    a = ap.parse_args(argv)

    current = sorted(glob.glob(a.current))
    baseline = sorted(glob.glob(a.baseline))
    if not current or not baseline:
        raise SystemExit(
            f"no CSVs matched (current={len(current)}, baseline={len(baseline)})"
        )

    result = compare_runs(
        current, baseline, threshold=a.threshold, min_cycles=a.min_cycles
    )
    report = render_report(
        result,
        threshold=a.threshold,
        min_cycles=a.min_cycles,
        test=a.test,
        baseline_sha=a.baseline_sha,
        current_sha=a.current_sha,
        baseline_iters=len(baseline),
        current_iters=len(current),
        baseline_label=a.baseline_label,
        current_label=a.current_label,
    )
    with open(a.report, "w") as f:
        f.write(report + "\n")

    stem = a.report.rsplit(".", 1)[0]
    written = [a.report]
    points = _points_csv(result["records"])
    if points is not None:
        points.to_csv(f"{stem}.points.csv", index=False)
        written.append(f"{stem}.points.csv")
    regressions = _points_csv(result["regressions"])
    if regressions is not None:
        regressions.to_csv(f"{stem}.regressions.csv", index=False)
        written.append(f"{stem}.regressions.csv")

    print(report)
    print("\n(wrote " + " + ".join(written) + ")")
    # exit non-zero if regressions, so the skill/CI can gate on it
    raise SystemExit(1 if result["regressions"] else 0)


if __name__ == "__main__":
    main()
