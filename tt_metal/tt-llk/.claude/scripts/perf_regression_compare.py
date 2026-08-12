# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained perf regression compare for the perf-regression-check skill.

Compares a branch's perf run (current) to the commit it was branched from
(baseline), for one test. Both sides may have several iterations; we take the
**median per point** on each side and flag points where current is more than
``threshold`` slower than baseline.

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


def compare_runs(current_csvs, baseline_csvs, *, threshold=0.05):
    """Median-vs-median comparison. Returns {records, regressions, new_points}.

    ``delta`` is the fractional change vs baseline (0.12 = 12% slower).
    """
    cur = _medians([pd.read_csv(p) for p in current_csvs])
    base = _medians([pd.read_csv(p) for p in baseline_csvs])

    records, regressions, new_points = [], [], []
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
        record = {
            **point,
            "baseline": bval,
            "delta": delta,
            "regression": delta > threshold,
        }
        records.append(record)
        if record["regression"]:
            regressions.append(record)
    return {"records": records, "regressions": regressions, "new_points": new_points}


_TOP_N = 25


def render_report(result, *, threshold, test, baseline_sha, current_sha, iters):
    """A short Markdown report: verdict + the worst regressions + new points.

    Only the ``_TOP_N`` biggest regressions are tabled; the full list goes to a
    companion ``*.regressions.csv`` (written by main).
    """
    regs = sorted(result["regressions"], key=lambda r: -r["delta"])
    verdict = "❌ REGRESSIONS FOUND" if regs else "✅ no regressions"
    lines = [
        f"# Perf regression check — {test}",
        "",
        f"**{verdict}** (threshold {threshold * 100:.0f}%, {iters} iteration(s)/side, median-vs-median)",
        "",
        f"- baseline (branch point on main): `{baseline_sha}`",
        f"- current (your branch HEAD): `{current_sha}`",
        f"- {len(result['records'])} points compared, "
        f"**{len(regs)} regression(s)**, {len(result['new_points'])} new point(s)",
        "",
    ]
    if regs:
        shown = regs[:_TOP_N]
        lines += [f"## Top {len(shown)} regressions (slower on your branch)", ""]
        lines += [
            "| marker | run type | current | baseline | Δ | config |",
            "|---|---|--:|--:|--:|---|",
        ]
        for r in shown:
            cfg = ", ".join(f"{k}={v}" for k, v in sorted(r["config"].items()))
            if len(cfg) > 90:
                cfg = cfg[:87] + "…"
            lines.append(
                f"| {r['marker']} | {r['run_type']} | {r['current']:.1f} | "
                f"{r['baseline']:.1f} | +{r['delta'] * 100:.1f}% | {cfg} |"
            )
        if len(regs) > _TOP_N:
            lines.append("")
            lines.append(
                f"_… and {len(regs) - _TOP_N} more — see the companion `.regressions.csv`._"
            )
        lines.append("")
    if result["new_points"]:
        lines.append(
            f"## New points ({len(result['new_points'])}) — no baseline, not counted as regressions"
        )
        lines.append(
            "_These configs/markers exist on your branch but not at the branch point._"
        )
    return "\n".join(lines)


def _write_regressions_csv(result, path):
    """Full regression list (every point, full config) as CSV — nothing truncated."""
    rows = []
    for r in sorted(result["regressions"], key=lambda x: -x["delta"]):
        row = {
            "marker": r["marker"],
            "run_type": r["run_type"],
            "current": r["current"],
            "baseline": r["baseline"],
            "delta_pct": round(r["delta"] * 100, 2),
        }
        row.update(r["config"])
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare current perf CSVs to a baseline's."
    )
    ap.add_argument("--current", required=True, help="glob for current-run CSVs")
    ap.add_argument("--baseline", required=True, help="glob for baseline-run CSVs")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--report", default="regression_report.md")
    ap.add_argument("--test", default="?")
    ap.add_argument("--baseline-sha", default="?")
    ap.add_argument("--current-sha", default="?")
    a = ap.parse_args(argv)

    current = sorted(glob.glob(a.current))
    baseline = sorted(glob.glob(a.baseline))
    if not current or not baseline:
        raise SystemExit(
            f"no CSVs matched (current={len(current)}, baseline={len(baseline)})"
        )

    result = compare_runs(current, baseline, threshold=a.threshold)
    report = render_report(
        result,
        threshold=a.threshold,
        test=a.test,
        baseline_sha=a.baseline_sha,
        current_sha=a.current_sha,
        iters=len(current),
    )
    with open(a.report, "w") as f:
        f.write(report + "\n")
    csv_path = a.report.rsplit(".", 1)[0] + ".regressions.csv"
    _write_regressions_csv(result, csv_path)
    print(report)
    print(
        f"\n(wrote {a.report}"
        + (f" + {csv_path}" if result["regressions"] else "")
        + ")"
    )
    # exit non-zero if regressions, so the skill/CI can gate on it
    raise SystemExit(1 if result["regressions"] else 0)


if __name__ == "__main__":
    main()
