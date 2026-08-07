#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Analyze D2HStreamService benchmark results.

Input: Google Benchmark JSON emitted by benchmark_d2h_stream_service
  (--benchmark_out=results.json --benchmark_out_format=json)

Benchmark name grammar:
  BM_D2HStreamService/<payload_regime>/<mode>/<host<serial|parallel>|threads<N>>/bytes<per_device_bytes>/pages<tensor_num_pages>/fifo_socket_pages<N>

The initial benchmark modes are:
  - size: throughput vs per-device payload size (serial + parallel host read)
  - page_granularity: throughput vs tensor page count for fixed payload sizes
  - host_threads: throughput vs host read thread count for fixed payload sizes

Usage:
  python3 analyze_d2h_stream_service.py run results.json
  python3 analyze_d2h_stream_service.py run results.json --out-dir analysis/
  python3 analyze_d2h_stream_service.py compare baseline.json candidate.json

Produces a normalized per-case CSV and line charts per family/mode. Compare mode
also emits a per-case delta CSV and baseline-vs-candidate overlays.
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
        "analyze_d2h_stream_service.py requires numpy and pandas. "
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

BENCHMARK_PREFIX = "BM_D2HStreamService"

MODE_AXIS = {
    "size": ("per_device_bytes", "per-device payload"),
    "page_granularity": ("tensor_num_pages", "tensor pages per device"),
    "host_threads": ("host_threads", "host read threads"),
    "socket_page_size": ("socket_page_bytes", "socket page size"),
}
INTEGER_SWEEP_AXES = (
    "tensor_num_pages",
    "fifo_socket_pages",
    "fifo_socket_pages_configured",
    "host_threads",
    "socket_page_bytes",
)

CASE_KEY = [
    "family",
    "mode",
    "host_read",
    "host_threads",
    "per_device_bytes",
    "tensor_num_pages",
    "socket_page_bytes",
    "fifo_socket_pages_configured",
]
PRIMARY_METRIC = "aggregate_gbps"
LATENCY_PLOT_METRICS = ("latency_p50_us", "latency_p90_us", "latency_max_us")
LATENCY_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
LATENCY_LINESTYLES = {
    "latency_p50_us": "-",
    "latency_p90_us": "--",
    "latency_max_us": ":",
}

AXIS_SHORT = {
    "per_device_bytes": "size",
    "tensor_num_pages": "pages",
    "tensor_page_bytes": "page_bytes",
    "fifo_socket_pages_configured": "fifo_socket_pages",
    "host_read": "host",
    "host_threads": "threads",
    "socket_page_bytes": "socket_page",
}

NUMERIC_COLUMNS = [
    "aggregate_gbps",
    "global_payload_gbps",
    "total_aggregate_bytes",
    "num_sockets",
    "warmup_iters",
    "perf_iters",
    "latency_iters",
    "per_device_bytes",
    "tensor_num_pages",
    "tensor_page_bytes",
    "target_socket_page_bytes",
    "max_socket_page_size_bytes",
    "socket_page_bytes",
    "fifo_socket_pages_configured",
    "fifo_size_bytes",
    "ack_worker_count",
    "parallel_host_read",
    "host_read_thread_count",
    "effective_host_read_thread_count",
    "host_threads",
    "per_shard_bytes",
    "socket_page_size",
    "num_socket_pages",
    "pages_per_chunk",
    "slot_count",
    "fifo_socket_pages",
    "fifo_transfer_depth",
    "host_fifo_depth_transfers",
    "pipeline_depth_transfers",
    "device_cb_depth_transfers",
    "barrier_tail_ms",
    "producer_finish_tail_ms",
    "latency_avg_us",
    "latency_p50_us",
    "latency_p90_us",
    "latency_max_us",
    "real_time",
    "cpu_time",
    "iterations",
]

_NAME_RE = re.compile(
    rf"^{re.escape(BENCHMARK_PREFIX)}/(?P<family>[^/]+)/(?P<mode>[^/]+)/"
    r"(?:(?:host(?P<host>serial|parallel))|(?:threads(?P<threads>\d+)))/"
    r"bytes(?P<bytes>\d+)/pages(?P<pages>\d+)"
    r"(?:/socket_page(?P<socket_page>\d+))?"
    r"/fifo_socket_pages(?P<fifo_socket_pages>\d+)"
)

_OLD_NAME_RE = re.compile(
    rf"^{re.escape(BENCHMARK_PREFIX)}/(?P<family>[^/]+)/(?P<mode>[^/]+)/"
    r"bytes(?P<bytes>\d+)/pages(?P<pages>\d+)/fifo_socket_pages(?P<fifo_socket_pages>\d+)"
)


def held_cols(mode: str, xcol: str) -> list[str]:
    if mode == "page_granularity":
        candidates = ("per_device_bytes", "fifo_socket_pages_configured")
    elif mode == "size":
        candidates = ("fifo_socket_pages_configured", "host_read")
    elif mode == "host_threads":
        candidates = ("per_device_bytes", "fifo_socket_pages_configured")
    elif mode == "socket_page_size":
        candidates = ("per_device_bytes", "fifo_socket_pages_configured")
    else:
        candidates = (
            "per_device_bytes",
            "tensor_num_pages",
            "tensor_page_bytes",
            "fifo_socket_pages_configured",
        )
    return [c for c in candidates if c != xcol]


def format_payload_bytes(value: float | int) -> str:
    value = int(value)
    mib = 1024 * 1024
    kib = 1024
    if value >= mib and value % mib == 0:
        return f"{value // mib} MB"
    if value >= kib and value % kib == 0:
        return f"{value // kib} KB"
    return f"{value} B"


def axis_value_label(column: str, value: float | int) -> str:
    if isinstance(value, str):
        return value
    if column in ("per_device_bytes", "tensor_page_bytes"):
        return format_payload_bytes(value)
    return str(int(value))


def line_label(held: list[str], key, varying: list[str]) -> str:
    key = key if isinstance(key, tuple) else (key,)
    parts = [f"{AXIS_SHORT[c]}={axis_value_label(c, v)}" for c, v in zip(held, key) if c in varying]
    return ", ".join(parts) if parts else "anchor"


def metric_label(metric: str) -> str:
    return {
        "aggregate_gbps": "aggregate D2H throughput (GB/s)",
        "global_payload_gbps": "global payload throughput (GB/s)",
        "latency_avg_us": "serialized release-to-read latency avg (us)",
        "latency_p50_us": "serialized release-to-read latency p50 (us)",
        "latency_p90_us": "serialized release-to-read latency p90 (us)",
        "latency_max_us": "serialized release-to-read latency max (us)",
    }.get(metric, metric)


def latency_percentile_label(metric: str) -> str:
    return metric.removeprefix("latency_").removesuffix("_us").upper()


def load_gbench_results(path: str | Path) -> pd.DataFrame:
    with open(path) as handle:
        payload = json.load(handle)
    df = pd.json_normalize(payload.get("benchmarks", []))
    if df.empty:
        return df
    if "run_type" in df.columns:
        df = df[df["run_type"] == "iteration"].copy()
    if "error_occurred" in df.columns:
        df = df[~df["error_occurred"].fillna(False).astype(bool)].copy()
    if "error_message" in df.columns:
        df = df[df["error_message"].isna() | (df["error_message"] == "")].copy()
    return df.reset_index(drop=True)


def parse_name(name: str) -> dict[str, str | int] | None:
    match = _NAME_RE.match(name)
    if match:
        host_threads = int(match.group("threads")) if match.group("threads") is not None else 0
        socket_page = int(match.group("socket_page")) if match.group("socket_page") is not None else 0
        return {
            "family": match.group("family"),
            "mode": match.group("mode"),
            "host_read": match.group("host") or "threads",
            "host_threads": host_threads,
            "per_device_bytes": int(match.group("bytes")),
            "tensor_num_pages": int(match.group("pages")),
            "socket_page_bytes": socket_page,
            "fifo_socket_pages_configured": int(match.group("fifo_socket_pages")),
        }
    match = _OLD_NAME_RE.match(name)
    if not match:
        return None
    return {
        "family": match.group("family"),
        "mode": match.group("mode"),
        "host_read": "parallel",
        "host_threads": 0,
        "per_device_bytes": int(match.group("bytes")),
        "tensor_num_pages": int(match.group("pages")),
        "socket_page_bytes": 0,
        "fifo_socket_pages_configured": int(match.group("fifo_socket_pages")),
    }


def normalize(path: str | Path) -> pd.DataFrame:
    raw = load_gbench_results(path)
    if raw.empty or "name" not in raw.columns:
        raise ValueError(f"No benchmark rows found in {path}")
    raw = raw.copy()
    parsed = raw["name"].apply(parse_name)
    raw = raw[parsed.notna()].copy()
    if raw.empty:
        raise ValueError(f"No {BENCHMARK_PREFIX} rows found in {path}")
    parsed_frame = pd.DataFrame(list(parsed.dropna()), index=raw.index)
    for col in parsed_frame.columns:
        raw[col] = parsed_frame[col]
    raw["benchmark_name"] = raw["name"]

    for col in NUMERIC_COLUMNS:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    missing = [c for c in CASE_KEY if c not in raw.columns]
    if missing:
        raise ValueError(f"Results missing required columns {missing}; is this the current benchmark?")

    agg = {col: ("median" if col in NUMERIC_COLUMNS else "first") for col in raw.columns if col not in CASE_KEY}
    df = raw.groupby(CASE_KEY, dropna=False).agg(agg).reset_index()
    df["source"] = str(Path(path).resolve())
    return df.sort_values(CASE_KEY).reset_index(drop=True)


def _save_fig(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def _set_sweep_axis(ax, values, *, xcol: str) -> None:
    vals = sorted({int(v) for v in values if np.isfinite(v)})
    if len(vals) <= 1:
        return
    if all(v > 0 for v in vals):
        ax.set_xscale("log", base=2)
    ax.set_xticks(vals)
    ax.set_xticklabels([axis_value_label(xcol, v) for v in vals])


def plot_family_lines(df: pd.DataFrame, out_dir: Path, prefix: str, metric: str = PRIMARY_METRIC) -> None:
    if not HAS_MATPLOTLIB:
        print("  Skipping plots: matplotlib is not installed")
        return
    if metric not in df.columns:
        return
    for (family, mode), sub in df.groupby(["family", "mode"]):
        sub = sub[sub[metric].notna()].copy()
        if sub.empty:
            continue
        xcol, xlabel = MODE_AXIS.get(mode, MODE_AXIS["size"])
        held = held_cols(mode, xcol)
        varying = [c for c in held if c in sub.columns and sub[c].nunique() > 1]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        if held:
            groups = sub.groupby(held)
        else:
            groups = [("anchor", sub)]
        for key, group in groups:
            group = group.sort_values(xcol)
            ax.plot(group[xcol], group[metric], marker="o", label=line_label(held, key, varying))
        _set_sweep_axis(ax, sub[xcol].values, xcol=xcol)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_label(metric))
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(f"{family}/{mode} sweep")
        if varying:
            ax.legend(title=", ".join(AXIS_SHORT[c] for c in varying), fontsize=8)
        _save_fig(fig, out_dir / f"{prefix}{family}_{mode}_{metric}.png")


def plot_compare_family_lines(merged: pd.DataFrame, out_dir: Path, prefix: str, metric: str = PRIMARY_METRIC) -> None:
    if not HAS_MATPLOTLIB:
        print("  Skipping compare plots: matplotlib is not installed")
        return
    base_col, cand_col = f"{metric}_baseline", f"{metric}_candidate"
    if base_col not in merged.columns or cand_col not in merged.columns:
        return
    shared = merged[(merged["_merge"] == "both") & merged[base_col].notna() & merged[cand_col].notna()].copy()
    if shared.empty:
        return
    cmap = plt.get_cmap("tab10")
    for (family, mode), sub in shared.groupby(["family", "mode"]):
        xcol, xlabel = MODE_AXIS.get(mode, MODE_AXIS["size"])
        held = held_cols(mode, xcol)
        varying = [c for c in held if c in sub.columns and sub[c].nunique() > 1]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        groups = sub.groupby(held) if held else [("anchor", sub)]
        for idx, (key, group) in enumerate(groups):
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
        _set_sweep_axis(ax, sub[xcol].values, xcol=xcol)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_label(metric))
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
        ax.set_title(f"{family}/{mode} sweep: baseline (solid) vs candidate (dashed)")
        _save_fig(fig, out_dir / f"{prefix}{family}_{mode}_{metric}.png")


def plot_latency_stat_lines(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if not HAS_MATPLOTLIB:
        print("  Skipping latency plots: matplotlib is not installed")
        return
    metrics = [metric for metric in LATENCY_PLOT_METRICS if metric in df.columns and df[metric].notna().any()]
    if not metrics:
        return
    latency_df = df[df[metrics].notna().any(axis=1)].copy()
    for (family, mode), sub in latency_df.groupby(["family", "mode"]):
        xcol, xlabel = MODE_AXIS.get(mode, MODE_AXIS["size"])
        held = held_cols(mode, xcol)
        varying = [c for c in held if c in sub.columns and sub[c].nunique() > 1]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        plotted = False
        groups = sub.groupby(held) if held else [("anchor", sub)]
        for line_idx, (key, group) in enumerate(groups):
            group = group.sort_values(xcol)
            base_label = line_label(held, key, varying)
            marker = LATENCY_MARKERS[line_idx % len(LATENCY_MARKERS)]
            for metric in metrics:
                metric_group = group[group[metric].notna()]
                if metric_group.empty:
                    continue
                label = latency_percentile_label(metric)
                if base_label != "anchor":
                    label = f"{base_label}, {label}"
                ax.plot(
                    metric_group[xcol],
                    metric_group[metric],
                    marker=marker,
                    linestyle=LATENCY_LINESTYLES.get(metric, "-"),
                    label=label,
                )
                plotted = True
        if not plotted:
            continue
        _set_sweep_axis(ax, sub[xcol].values, xcol=xcol)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("serialized release-to-read latency (us)")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(f"{family}/{mode} serialized latency stats")
        ax.legend(fontsize=8)
        _save_fig(fig, out_dir / f"{prefix}{family}_{mode}_latency_stats_us.png")


def plot_compare_latency_stat_lines(merged: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if not HAS_MATPLOTLIB:
        print("  Skipping compare latency plots: matplotlib is not installed")
        return
    metrics = [
        metric
        for metric in LATENCY_PLOT_METRICS
        if f"{metric}_baseline" in merged.columns and f"{metric}_candidate" in merged.columns
    ]
    if not metrics:
        return
    shared = merged[merged["_merge"] == "both"].copy()
    if shared.empty:
        return
    cmap = plt.get_cmap("tab10")
    for (family, mode), sub in shared.groupby(["family", "mode"]):
        xcol, xlabel = MODE_AXIS.get(mode, MODE_AXIS["size"])
        held = held_cols(mode, xcol)
        varying = [c for c in held if c in sub.columns and sub[c].nunique() > 1]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        plotted = False
        color_idx = 0
        groups = sub.groupby(held) if held else [("anchor", sub)]
        for line_idx, (key, group) in enumerate(groups):
            group = group.sort_values(xcol)
            base_label = line_label(held, key, varying)
            marker = LATENCY_MARKERS[line_idx % len(LATENCY_MARKERS)]
            for metric in metrics:
                base_col, cand_col = f"{metric}_baseline", f"{metric}_candidate"
                metric_group = group[group[base_col].notna() & group[cand_col].notna()]
                if metric_group.empty:
                    continue
                label = latency_percentile_label(metric)
                if base_label != "anchor":
                    label = f"{base_label}, {label}"
                color = cmap(color_idx % 10)
                ax.plot(
                    metric_group[xcol],
                    metric_group[base_col],
                    marker=marker,
                    linestyle="-",
                    color=color,
                    label=f"{label} baseline",
                )
                ax.plot(
                    metric_group[xcol],
                    metric_group[cand_col],
                    marker=marker,
                    linestyle="--",
                    color=color,
                    label=f"{label} candidate",
                )
                plotted = True
                color_idx += 1
        if not plotted:
            continue
        _set_sweep_axis(ax, sub[xcol].values, xcol=xcol)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("serialized release-to-read latency (us)")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
        ax.set_title(f"{family}/{mode} serialized latency stats: baseline (solid) vs candidate (dashed)")
        _save_fig(fig, out_dir / f"{prefix}{family}_{mode}_latency_stats_us.png")


def _case_id(row) -> str:
    host_part = f"threads={int(row.host_threads)}" if row.host_read == "threads" else f"host={row.host_read}"
    return (
        f"{row.family}/{row.mode}/{host_part}/size={format_payload_bytes(row.per_device_bytes)}/"
        f"pages={int(row.tensor_num_pages)}/fifo_socket_pages={int(row.fifo_socket_pages_configured)}"
    )


def print_run_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print(f"  D2HStreamService benchmark summary ({len(df)} case row(s))")
    print("=" * 72)
    peak = df.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]
    print(f"  Peak aggregate throughput: {peak[PRIMARY_METRIC]:.3f} GB/s ({_case_id(peak)})")

    for (family, mode), sub in df.groupby(["family", "mode"]):
        xcol, xlabel = MODE_AXIS.get(mode, MODE_AXIS["size"])
        held = held_cols(mode, xcol)
        varying = [c for c in held if c in sub.columns and sub[c].nunique() > 1]
        groups = sub.groupby(held) if held else [("anchor", sub)]
        print(f"\n  [{family}/{mode}] {PRIMARY_METRIC} vs {xlabel} ({len(list(groups))} line(s)):")
        groups = sub.groupby(held) if held else [("anchor", sub)]
        for key, group in groups:
            group = group.sort_values(xcol)
            lo, hi = group.iloc[0], group.iloc[-1]
            label = line_label(held, key, varying)
            print(
                f"    {label:<28} {group[PRIMARY_METRIC].min():6.3f}-{group[PRIMARY_METRIC].max():6.3f} GB/s  "
                f"[{AXIS_SHORT[xcol]} {axis_value_label(xcol, lo[xcol])}->{axis_value_label(xcol, hi[xcol])}: "
                f"{lo[PRIMARY_METRIC]:.3f} -> {hi[PRIMARY_METRIC]:.3f}]"
            )

    tail_cols = [c for c in ("barrier_tail_ms", "producer_finish_tail_ms") if c in df.columns]
    if tail_cols:
        print("\n  Untimed tail diagnostics:")
        for col in tail_cols:
            values = df[col].dropna()
            if not values.empty:
                print(f"    {col}: median={values.median():.3f} ms, max={values.max():.3f} ms")


def write_run_outputs(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / f"{prefix}d2h_stream_service_cases.csv"
    df.to_csv(cases_path, index=False, float_format="%.6f")
    print(f"  Saved {cases_path}")
    plot_family_lines(df, out_dir, prefix)
    plot_latency_stat_lines(df, out_dir, prefix)


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


def print_compare_report(merged: pd.DataFrame) -> None:
    shared = merged[merged["_merge"] == "both"]
    base_only = merged[merged["_merge"] == "left_only"]
    cand_only = merged[merged["_merge"] == "right_only"]

    print("\n" + "=" * 72)
    print("  D2HStreamService benchmark comparison")
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
        return f"    {_case_id(row):<48} {row[pct]:+7.2f}%  ({row[base_col]:.3f} -> {row[cand_col]:.3f} GB/s)"

    print("\n  Top improvements:")
    for _, row in ranked.head(10).iterrows():
        print(fmt(row))
    print("\n  Top regressions:")
    for _, row in ranked.tail(10).iloc[::-1].iterrows():
        print(fmt(row))


def write_compare_outputs(merged: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_path = out_dir / f"{prefix}d2h_stream_service_compare.csv"
    merged.to_csv(compare_path, index=False, float_format="%.6f")
    print(f"  Saved {compare_path}")
    plot_compare_family_lines(merged, out_dir, prefix)
    plot_compare_latency_stat_lines(merged, out_dir, prefix)


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
    parser = argparse.ArgumentParser(description="Analyze D2HStreamService benchmark results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Analyze a single Google Benchmark JSON")
    run_parser.add_argument("results", help="Path to benchmark_d2h_stream_service results JSON")
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
