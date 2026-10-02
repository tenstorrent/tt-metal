# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare two perf runs point by point and report the regressions.

Run it by path: ``tt_metal/tt-llk`` is not an importable package.
"""

import argparse
import csv
import glob
import json
import os
import re

import pandas as pd

# 5 runs of one commit on Blackhole moved at most 1.9% and 25 cycles (#53752).
DEFAULT_THRESHOLD = 0.02
DEFAULT_MIN_CYCLES = 30.0

# Same names as helpers/perf/core.py, which this file cannot import.
RUN_TYPE_PRESETS = {
    "ALL_ISOLATION_MODES": ("UNPACK_ISOLATE", "MATH_ISOLATE", "PACK_ISOLATE"),
    "ALL_MODES": ("L1_TO_L1", "UNPACK_ISOLATE", "MATH_ISOLATE", "PACK_ISOLATE"),
}


def _module_thresholds(spec):
    """``perf_matmul=0.25,perf_math_matmul=0.25`` -> {module: threshold}."""
    out = {}
    for item in filter(None, (s.strip() for s in (spec or "").split(","))):
        module, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(
                f"--module-threshold: expected module=threshold, got {item!r}"
            )
        out[module.strip()] = float(value)
    return out


def _is_metric(col):
    return col.startswith("mean(") or col.startswith("std(")


MODULE_COL = "test_module"


def _is_ignored(col):
    # Not part of a point's identity: timing stats and code size.
    return _is_metric(col) or col.startswith("TEXT_SIZE(")


def _run_type_of(mean_col):
    """``mean(L1_TO_L1)`` -> ``L1_TO_L1``."""
    return mean_col[len("mean(") : -1]


def _filter_run_types(medians, allowed):
    """Keep only the metrics whose run type was asked for."""
    return {k: v for k, v in medians.items() if _run_type_of(k[1]) in allowed}


def _mean_cols(columns):
    return [c for c in columns if c.startswith("mean(")]


def _config_cols(columns):
    return [c for c in columns if c != "marker" and not _is_ignored(c)]


_PAIR_POOL = {}


def _pair(pair):
    """One shared object per distinct (column, value); there are only thousands."""
    return _PAIR_POOL.setdefault(pair, pair)


def _point_key(marker, config_pairs):
    """A point's identity within one test: marker + the non-null sweep config."""
    return (
        marker,
        tuple(sorted(_pair((c, v)) for c, v in config_pairs if pd.notna(v))),
    )


def _medians(frames):
    """{(point_key, mean_col): median value} across a list of run DataFrames.

    One groupby over the concatenated frames, rather than a Python loop over
    every row of every iteration. A full sweep is tens of thousands of rows and
    the gate reads both sides of it.
    """
    frames = [f for f in frames if not f.empty]
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    if "marker" not in df.columns:
        df = df.copy()
        df["marker"] = pd.NA
    mean_cols = _mean_cols(df.columns)
    if not mean_cols:
        return {}
    # Drop all-null config columns so they stay out of the key, exactly as the
    # per-row null skip did, and so groupby is not handed an empty axis.
    config_cols = [c for c in _config_cols(df.columns) if df[c].notna().any()]
    grouped = df.groupby(["marker", *config_cols], dropna=False, sort=False)[
        mean_cols
    ].median()

    out = {}
    id_frame = grouped.reset_index()
    n_id = 1 + len(config_cols)
    for rec in id_frame.itertuples(index=False, name=None):
        key = _point_key(rec[0], zip(config_cols, rec[1:n_id]))
        for col, val in zip(mean_cols, rec[n_id:]):
            if pd.notna(val):
                out[(key, col)] = float(val)
    return out


def _module_of(path):
    """perf_x[.post|.counters].csv -> perf_x, the rule the warehouse uses for TEST_NAME."""
    return re.sub(r"\.(?:post|counters)?\.?csv$", "", os.path.basename(path)).rstrip(
        "."
    )


def _read_frames(paths):
    """Each side's frames, stamped with their test module unless they carry one."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if MODULE_COL not in frame.columns:
            frame[MODULE_COL] = _module_of(path)
        frames.append(frame)
    return frames


def compare_runs(
    current_csvs,
    baseline_csvs,
    *,
    threshold=DEFAULT_THRESHOLD,
    min_cycles=DEFAULT_MIN_CYCLES,
    run_types=None,
    module_thresholds=None,
):
    """Median-vs-median comparison.

    Returns ``{records, regressions, improvements, new_points, noise_filtered}``.
    ``delta`` is the fractional change vs baseline (0.12 = 12% slower, -0.12 = 12%
    faster) and ``abs_delta`` is the same change in cycles. A regression needs
    ``delta > threshold`` AND ``abs_delta > min_cycles``; an improvement is the
    mirror image. See the constants above for why both are required.

    ``noise_filtered`` counts points that cleared the percentage but not the cycle
    floor — exactly the points a relative-only rule would have failed on.

    If ``run_types`` is specified (comma-separated, e.g. "L1_TO_L1,MATH_ISOLATE"),
    only compare metrics for those run types.
    """
    cur = _medians(_read_frames(current_csvs))
    base = _medians(_read_frames(baseline_csvs))

    if run_types:
        allowed = set()
        for name in (t.strip() for t in run_types.split(",")):
            allowed.update(RUN_TYPE_PRESETS.get(name, (name,)))
        cur = _filter_run_types(cur, allowed)
        base = _filter_run_types(base, allowed)

    records, regressions, improvements, new_points = [], [], [], []
    noise_filtered = 0
    for (key, mean_col), cval in cur.items():
        marker, config = key
        run_type = _run_type_of(mean_col)
        point = {
            "marker": marker,
            "run_type": run_type,
            "config": config,
            "current": cval,
            MODULE_COL: dict(config).get(MODULE_COL, ""),
        }
        bval = base.get((key, mean_col))
        if bval is None:
            new_points.append(point)  # no baseline (new config / new test)
            continue
        limit = (module_thresholds or {}).get(point[MODULE_COL], threshold)
        delta = (cval - bval) / bval if bval else 0.0
        abs_delta = cval - bval
        big_enough = abs(abs_delta) > min_cycles
        if abs(delta) > limit and not big_enough:
            noise_filtered += 1
        record = {
            **point,
            "baseline": bval,
            "delta": delta,
            "abs_delta": abs_delta,
            "regression": delta > limit and big_enough,
            "improvement": delta < -limit and big_enough,
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
    tail = f" — {iters} CSV file(s)" if iters else ""
    return f"- {named}: `{sha}`{tail}"


def _varying_keys(rows):
    """Config keys that differ across the rows shown; those tell them apart."""
    keys = {k for r in rows for k, _ in r["config"] if k != MODULE_COL}
    return {k for k in keys if len({dict(r["config"]).get(k) for r in rows}) > 1}


def _findings(rows):
    """Points that differ only in marker or run type are one finding."""
    groups = {}
    for r in rows:
        groups.setdefault((r.get(MODULE_COL, ""), r["config"]), []).append(r)
    return sorted(groups.values(), key=lambda g: -max(abs(p["delta"]) for p in g))


def _delta_table(rows, *, caption):
    """One line per finding, then each finding's configuration in full."""
    groups = _findings(rows)
    varying = _varying_keys([g[0] for g in groups])
    lines = [
        f"## {caption}",
        "",
        "| # | test | points | markers | run types | current | baseline | Δ | Δ cycles |",
        "|--:|---|--:|---|---|--:|--:|--:|--:|",
    ]
    for n, g in enumerate(groups, 1):
        w = max(g, key=lambda p: abs(p["delta"]))
        markers = ", ".join(sorted({p["marker"] for p in g}))
        run_types = ", ".join(sorted({p["run_type"] for p in g}))
        lines.append(
            f"| {n} | {w.get(MODULE_COL) or '?'} | {len(g)} | {markers} | {run_types} | "
            f"{w['current']:.1f} | {w['baseline']:.1f} | {w['delta'] * 100:+.1f}% | "
            f"{w.get('abs_delta', 0.0):+.0f} |"
        )
    lines += ["", "### Configuration of each row above", ""]
    for n, g in enumerate(groups, 1):
        config = {k: v for k, v in g[0]["config"] if k != MODULE_COL}
        ordered = sorted(config, key=lambda k: (k not in varying, k))
        lines.append(f"{n}. " + ", ".join(f"`{k}={config[k]}`" for k in ordered))
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
    module_thresholds=None,
):
    """Markdown: the verdict, the worst findings each way, and the new points."""
    regs = sorted(result["regressions"], key=lambda r: -r["delta"])
    imps = sorted(result.get("improvements", []), key=lambda r: r["delta"])
    reg_groups, imp_groups = _findings(regs), _findings(imps)
    verdict = "❌ REGRESSIONS FOUND" if regs else "✅ no regressions"
    lines = [
        f"# Perf compare — {test}",
        "",
        f"**{verdict}**",
        "",
        f"Rule: a point is a regression when it is **more than {threshold * 100:.0f}% "
        f"slower AND more than {min_cycles:.0f} cycles slower**. Both must hold. "
        "Comparison is median-vs-median, per (marker, run type, sweep config)."
        + (
            " Per-module thresholds: "
            + ", ".join(
                f"{m} {v * 100:.0f}%" for m, v in sorted(module_thresholds.items())
            )
            + "."
            if module_thresholds
            else ""
        ),
        "",
        _side_line("baseline", baseline_sha, baseline_label, baseline_iters),
        _side_line("current", current_sha, current_label, current_iters),
        f"- {len(result['records'])} points compared, "
        f"**{len(regs)} regressed point(s) in {len(reg_groups)} finding(s)**, "
        f"{len(imps)} improved point(s) in {len(imp_groups)} finding(s), "
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
    for groups, what, where, companion in (
        (reg_groups, "regression", "slower", ".regressions.csv"),
        (imp_groups, "improvement", "faster", ".points.csv"),
    ):
        if not groups:
            continue
        shown = groups[:_TOP_N]
        lines += _delta_table(
            [p for g in shown for p in g],
            caption=f"Top {len(shown)} {what} findings ({where} on current)",
        )
        if len(groups) > _TOP_N:
            lines += ["", f"_… and {len(groups) - _TOP_N} more — see `{companion}`._"]
        lines.append("")
    if result["new_points"]:
        lines.append(
            f"## New points ({len(result['new_points'])}) — no baseline, not counted as regressions"
        )
        lines.append(
            "_These configs/markers exist at the current commit but not at the baseline commit._"
        )
    return "\n".join(lines)


def _write_points_csv(records, path):
    """Stream records to CSV, worst delta first. One row is held at a time."""
    if not records:
        return False
    fixed = [
        "marker",
        "run_type",
        "current",
        "baseline",
        "delta_pct",
        "delta_cycles",
        MODULE_COL,
    ]
    config_cols = sorted(
        {k for r in records for k, _ in r["config"] if k != MODULE_COL}
    )
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fixed + config_cols)
        writer.writeheader()
        for r in sorted(records, key=lambda x: -x["delta"]):
            row = {
                "marker": r["marker"],
                "run_type": r["run_type"],
                "current": r["current"],
                "baseline": r["baseline"],
                "delta_pct": round(r["delta"] * 100, 2),
                "delta_cycles": round(r.get("abs_delta", 0.0), 1),
                MODULE_COL: r.get(MODULE_COL, ""),
            }
            row.update(r["config"])
            writer.writerow(row)
    return True


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
    ap.add_argument(
        "--run-types",
        default=None,
        help="comma-separated run types to compare (e.g. L1_TO_L1,MATH_ISOLATE). "
        "if not specified, all run types are compared.",
    )
    ap.add_argument(
        "--module-threshold",
        default="",
        help="per-module thresholds, e.g. perf_matmul=0.25,perf_math_matmul=0.25",
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

    current = sorted(glob.glob(a.current, recursive=True))
    baseline = sorted(glob.glob(a.baseline, recursive=True))

    # Exclude .post.csv files (postprocessed versions halve the cycle counts)
    current = [p for p in current if not p.endswith(".post.csv")]
    baseline = [p for p in baseline if not p.endswith(".post.csv")]

    if not current or not baseline:
        raise SystemExit(
            f"no CSVs matched (current={len(current)}, baseline={len(baseline)})"
        )

    result = compare_runs(
        current,
        baseline,
        threshold=a.threshold,
        min_cycles=a.min_cycles,
        run_types=a.run_types,
        module_thresholds=_module_thresholds(a.module_threshold),
    )

    # Fail if zero points were compared (e.g., all-new configs, run-type filter mismatch)
    if not result["records"]:
        print(
            f"❌ No points compared. "
            f"Check: run-types filter matches CSV content, run-types={a.run_types}"
        )
        raise SystemExit(1)

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
        module_thresholds=_module_thresholds(a.module_threshold),
    )
    with open(a.report, "w") as f:
        f.write(report + "\n")

    stem = a.report.rsplit(".", 1)[0]
    written = [a.report]
    if _write_points_csv(result["records"], f"{stem}.points.csv"):
        written.append(f"{stem}.points.csv")
    if _write_points_csv(result["regressions"], f"{stem}.regressions.csv"):
        written.append(f"{stem}.regressions.csv")

    print(report)
    print("\n(wrote " + " + ".join(written) + ")")
    # Written last, so its absence means the comparison died rather than ran.
    with open(f"{stem}.verdict.json", "w") as fh:
        json.dump(
            {
                "status": "regressed" if result["regressions"] else "clean",
                "regressions": len(result["regressions"]),
                "points": len(result["records"]),
            },
            fh,
        )
    # exit non-zero if regressions, so the skill/CI can gate on it
    raise SystemExit(1 if result["regressions"] else 0)


if __name__ == "__main__":
    main()
