#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Analyze H2DStreamService benchmark results.

Input: Google Benchmark JSON emitted by benchmark_h2d_stream_service
  (--benchmark_out=results.json --benchmark_out_format=json)

The benchmark sweeps three independent 1D families (all FullShard2D, distributed-tensor
input, fixed 4x4 worker grid), each holding the other chunk-plan axis at a shared anchor:
  - size: throughput vs per_device_pages   (cb/fifo fixed at the anchor)
  - cb:   throughput vs cb_pages            (fifo fixed at the anchor)
  - fifo: throughput vs fifo_pages          (cb fixed at the anchor)

Benchmark name grammar: BM_H2DStreamService/<family>/p<pages>/cb<cb>/fifo<fifo>

Usage:
  python3 analyze_h2d_stream_service.py run results.json
  python3 analyze_h2d_stream_service.py run results.json --out-dir analysis/
  python3 analyze_h2d_stream_service.py compare baseline.json candidate.json

Produces a normalized per-case CSV and one throughput line chart per family (plus
baseline-vs-candidate overlays and a per-case delta CSV in compare mode).

Note on metrics: these cases are fully sharded (FullShard2D), so aggregate feeder
throughput equals global payload throughput; charts use aggregate_gbps and the CSV
carries both.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "analyze_h2d_stream_service.py requires numpy and pandas. "
        "Run it from the project Python environment or install the missing packages. "
        f"Missing module: {exc.name}"
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False

BENCHMARK_PREFIX = "BM_H2DStreamService"

# family -> (swept-axis column, human-readable axis label)
FAMILY_AXIS = {
    "size": ("per_device_pages", "per-device pages"),
    "cb": ("cb_pages", "scratch-CB pages"),
    "fifo": ("fifo_pages", "FIFO pages"),
}

CASE_KEY = ["family", "per_device_pages", "cb_pages", "fifo_pages"]
PRIMARY_METRIC = "aggregate_gbps"

# Short names for the three sweepable axes, used in per-line labels.
AXIS_SHORT = {"per_device_pages": "pages", "cb_pages": "cb", "fifo_pages": "fifo"}


def held_cols(xcol: str) -> list[str]:
    """The two axes a family holds (or perturbs across lines) while sweeping `xcol`."""
    return [c for c in ("per_device_pages", "cb_pages", "fifo_pages") if c != xcol]


def line_label(held: list[str], key, varying: list[str]) -> str:
    """Label a line by the held-axis values that actually vary within the family."""
    key = key if isinstance(key, tuple) else (key,)
    parts = [f"{AXIS_SHORT[c]}={int(v)}" for c, v in zip(held, key) if c in varying]
    return ", ".join(parts) if parts else "anchor"


NUMERIC_COLUMNS = [
    "aggregate_gbps",
    "global_payload_gbps",
    "total_aggregate_bytes",
    "num_sockets",
    "warmup_iters",
    "perf_iters",
    "per_device_pages",
    "cb_pages",
    "fifo_pages",
    "fifo_size_bytes",
    "scratch_cb_size_bytes",
    "worker_count",
    "per_shard_bytes",
    "socket_page_size",
    "num_socket_pages",
    "pages_per_chunk",
    "real_time",
    "cpu_time",
    "iterations",
]

# Tolerates the trailing "/manual_time" that UseManualTime() appends to the name.
_NAME_RE = re.compile(
    rf"^{re.escape(BENCHMARK_PREFIX)}/(?P<family>[^/]+)/p(?P<pages>\d+)/cb(?P<cb>\d+)/fifo(?P<fifo>\d+)"
)


def metric_label(metric: str) -> str:
    return {
        "aggregate_gbps": "aggregate feeder throughput (GB/s)",
        "global_payload_gbps": "global payload throughput (GB/s)",
    }.get(metric, metric)


def load_gbench_results(path: str | Path) -> pd.DataFrame:
    with open(path) as handle:
        payload = json.load(handle)
    df = pd.json_normalize(payload.get("benchmarks", []))
    if df.empty:
        return df
    # Keep only per-iteration rows; drop GBench's _mean/_median/_stddev aggregate rows
    # (emitted under --benchmark_repetitions) so they don't pollute the per-case median.
    if "run_type" in df.columns:
        df = df[df["run_type"] == "iteration"].copy()
    if "error_occurred" in df.columns:
        df = df[~df["error_occurred"].fillna(False).astype(bool)].copy()
    if "error_message" in df.columns:
        df = df[df["error_message"].isna() | (df["error_message"] == "")].copy()
    return df.reset_index(drop=True)


def parse_family(name: str) -> str | None:
    match = _NAME_RE.match(name)
    return match.group("family") if match else None


def normalize(path: str | Path) -> pd.DataFrame:
    raw = load_gbench_results(path)
    if raw.empty or "name" not in raw.columns:
        raise ValueError(f"No benchmark rows found in {path}")
    raw = raw.copy()
    raw["family"] = raw["name"].apply(parse_family)
    raw = raw[raw["family"].notna()].copy()
    if raw.empty:
        raise ValueError(f"No {BENCHMARK_PREFIX} rows found in {path}")
    raw["benchmark_name"] = raw["name"]

    for col in NUMERIC_COLUMNS:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    missing = [c for c in CASE_KEY if c not in raw.columns]
    if missing:
        raise ValueError(f"Results missing required columns {missing}; is this the current benchmark?")

    # Median across any per-repetition rows for the same case.
    agg = {col: ("median" if col in NUMERIC_COLUMNS else "first") for col in raw.columns if col not in CASE_KEY}
    df = raw.groupby(CASE_KEY, dropna=False).agg(agg).reset_index()
    df["source"] = str(Path(path).resolve())
    return df.sort_values(CASE_KEY).reset_index(drop=True)


def _save_fig(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def _set_log2_axis(ax, values) -> None:
    vals = sorted({int(v) for v in values if np.isfinite(v)})
    if len(vals) > 1 and all(v > 0 for v in vals):
        ax.set_xscale("log", base=2)
        ax.set_xticks(vals)
        ax.set_xticklabels([str(v) for v in vals])


def plot_family_lines(df: pd.DataFrame, out_dir: Path, prefix: str, metric: str = PRIMARY_METRIC) -> None:
    if not HAS_MATPLOTLIB:
        print("  Skipping plots: matplotlib is not installed")
        return
    for family, (xcol, xlabel) in FAMILY_AXIS.items():
        sub = df[df["family"] == family]
        if sub.empty:
            continue
        held = held_cols(xcol)
        varying = [c for c in held if sub[c].nunique() > 1]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for key, group in sub.groupby(held):
            group = group.sort_values(xcol)
            ax.plot(group[xcol], group[metric], marker="o", label=line_label(held, key, varying))
        _set_log2_axis(ax, sub[xcol].values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_label(metric))
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(f"{family} sweep")
        if varying:
            ax.legend(title=", ".join(AXIS_SHORT[c] for c in varying), fontsize=8)
        _save_fig(fig, out_dir / f"{prefix}{family}_{metric}.png")


def plot_compare_family_lines(merged: pd.DataFrame, out_dir: Path, prefix: str, metric: str = PRIMARY_METRIC) -> None:
    if not HAS_MATPLOTLIB:
        print("  Skipping compare plots: matplotlib is not installed")
        return
    base_col, cand_col = f"{metric}_baseline", f"{metric}_candidate"
    if base_col not in merged.columns or cand_col not in merged.columns:
        return
    shared = merged[merged["_merge"] == "both"]
    cmap = plt.get_cmap("tab10")
    for family, (xcol, xlabel) in FAMILY_AXIS.items():
        sub = shared[shared["family"] == family]
        if sub.empty:
            continue
        held = held_cols(xcol)
        varying = [c for c in held if sub[c].nunique() > 1]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for idx, (key, group) in enumerate(sub.groupby(held)):
            group = group.sort_values(xcol)
            color = cmap(idx % 10)
            ax.plot(
                group[xcol],
                group[base_col],
                marker="o",
                linestyle="-",
                color=color,
                label=line_label(held, key, varying),
            )
            ax.plot(group[xcol], group[cand_col], marker="s", linestyle="--", color=color)
        _set_log2_axis(ax, sub[xcol].values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_label(metric))
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
        ax.set_title(f"{family} sweep: baseline (solid) vs candidate (dashed)")
        _save_fig(fig, out_dir / f"{prefix}{family}_{metric}.png")


def print_run_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print(f"  H2DStreamService benchmark summary ({len(df)} case row(s))")
    print("=" * 72)
    peak = df.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]
    print(f"  Peak aggregate throughput: {peak[PRIMARY_METRIC]:.3f} GB/s ({peak.benchmark_name})")

    for family, (xcol, xlabel) in FAMILY_AXIS.items():
        sub = df[df["family"] == family]
        if sub.empty:
            continue
        held = held_cols(xcol)
        varying = [c for c in held if sub[c].nunique() > 1]
        print(f"\n  [{family}] {PRIMARY_METRIC} vs {xlabel} ({len(list(sub.groupby(held)))} line(s)):")
        for key, group in sub.groupby(held):
            group = group.sort_values(xcol)
            lo, hi = group.iloc[0], group.iloc[-1]
            label = line_label(held, key, varying)
            print(
                f"    {label:<22} {group[PRIMARY_METRIC].min():6.3f}-{group[PRIMARY_METRIC].max():6.3f} GB/s  "
                f"[{AXIS_SHORT[xcol]} {int(lo[xcol])}->{int(hi[xcol])}: "
                f"{lo[PRIMARY_METRIC]:.3f} -> {hi[PRIMARY_METRIC]:.3f}]"
            )

    # Configs measured under more than one family meet at the shared anchor: a consistency check.
    multi = df.groupby(["per_device_pages", "cb_pages", "fifo_pages"]).filter(lambda g: g["family"].nunique() > 1)
    if not multi.empty:
        print("\n  Cross-check (same config measured under multiple families):")
        for (pages, cb, fifo), g in multi.groupby(["per_device_pages", "cb_pages", "fifo_pages"]):
            readings = "  ".join(f"{r.family}={r[PRIMARY_METRIC]:.3f}" for _, r in g.iterrows())
            print(f"    p{int(pages)}/cb{int(cb)}/fifo{int(fifo)}: {readings} GB/s")


def write_run_outputs(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / f"{prefix}h2d_stream_service_cases.csv"
    df.to_csv(cases_path, index=False, float_format="%.6f")
    print(f"  Saved {cases_path}")
    plot_family_lines(df, out_dir, prefix)


def compare_runs(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    merged = baseline_df.merge(
        candidate_df, on=CASE_KEY, suffixes=("_baseline", "_candidate"), how="outer", indicator=True
    )
    base_col, cand_col = f"{PRIMARY_METRIC}_baseline", f"{PRIMARY_METRIC}_candidate"
    if base_col in merged.columns and cand_col in merged.columns:
        merged[f"{PRIMARY_METRIC}_delta_abs"] = merged[cand_col] - merged[base_col]
        merged[f"{PRIMARY_METRIC}_delta_pct"] = (
            100.0 * merged[f"{PRIMARY_METRIC}_delta_abs"] / merged[base_col].replace(0, np.nan)
        )
    return merged.sort_values(CASE_KEY).reset_index(drop=True)


def _case_id(row) -> str:
    return f"{row.family}/p{int(row.per_device_pages)}/cb{int(row.cb_pages)}/fifo{int(row.fifo_pages)}"


def print_compare_report(merged: pd.DataFrame) -> None:
    shared = merged[merged["_merge"] == "both"]
    base_only = merged[merged["_merge"] == "left_only"]
    cand_only = merged[merged["_merge"] == "right_only"]

    print("\n" + "=" * 72)
    print("  H2DStreamService benchmark comparison")
    print("=" * 72)
    print(f"  Shared cases   : {len(shared)}")
    print(f"  Baseline only  : {len(base_only)}")
    print(f"  Candidate only : {len(cand_only)}")
    for label, frame in (("baseline", base_only), ("candidate", cand_only)):
        if not frame.empty:
            print(f"\n  Cases only in {label}:")
            for _, row in frame.iterrows():
                print(f"    {_case_id(row)}")

    pct = f"{PRIMARY_METRIC}_delta_pct"
    if pct not in shared.columns or shared.empty:
        return
    ranked = shared.sort_values(pct, ascending=False)
    base_col, cand_col = f"{PRIMARY_METRIC}_baseline", f"{PRIMARY_METRIC}_candidate"

    def fmt(row) -> str:
        return f"    {_case_id(row):<28} {row[pct]:+7.2f}%  ({row[base_col]:.3f} -> {row[cand_col]:.3f} GB/s)"

    print("\n  Top improvements:")
    for _, row in ranked.head(10).iterrows():
        print(fmt(row))
    print("\n  Top regressions:")
    for _, row in ranked.tail(10).iloc[::-1].iterrows():
        print(fmt(row))


def write_compare_outputs(merged: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_path = out_dir / f"{prefix}h2d_stream_service_compare.csv"
    merged.to_csv(compare_path, index=False, float_format="%.6f")
    print(f"  Saved {compare_path}")
    plot_compare_family_lines(merged, out_dir, prefix)


def default_out_dir(path: str | Path) -> Path:
    return Path(path).resolve().parent


def run_mode(results: str, out_dir: str | None, prefix: str) -> None:
    df = normalize(results)
    print_run_report(df)
    write_run_outputs(df, Path(out_dir) if out_dir else default_out_dir(results), prefix)


def compare_mode(baseline: str, candidate: str, out_dir: str | None, prefix: str) -> None:
    merged = compare_runs(normalize(baseline), normalize(candidate))
    print_compare_report(merged)
    write_compare_outputs(merged, Path(out_dir) if out_dir else default_out_dir(candidate), prefix)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze H2DStreamService benchmark results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Analyze a single Google Benchmark JSON")
    run_parser.add_argument("results", help="Path to benchmark_h2d_stream_service results JSON")
    run_parser.add_argument("--out-dir", default=None, help="Directory for generated CSV and plots")
    run_parser.add_argument("--prefix", default="", help="Prefix prepended to every output file name")

    compare_parser = subparsers.add_parser("compare", help="Compare baseline and candidate JSONs")
    compare_parser.add_argument("baseline", help="Baseline benchmark results JSON")
    compare_parser.add_argument("candidate", help="Candidate benchmark results JSON")
    compare_parser.add_argument("--out-dir", default=None, help="Directory for generated CSV and plots")
    compare_parser.add_argument("--prefix", default="compare_", help="Prefix prepended to every output file name")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "run":
        run_mode(args.results, args.out_dir, args.prefix)
    elif args.command == "compare":
        compare_mode(args.baseline, args.candidate, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()
