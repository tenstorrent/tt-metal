#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Analyze H2D/D2H socket benchmark results.

Input: Google Benchmark CSV from distributed_benchmarks
  (--benchmark_format=csv --benchmark_out=results.csv)

Usage:
  python3 analyze_hd_sockets.py --gbench          results.csv   # auto-detect type
  python3 analyze_hd_sockets.py --d2h-throughput   results.csv
  python3 analyze_hd_sockets.py --d2h-latency      results.csv
  python3 analyze_hd_sockets.py --d2h-ping         ping_dir/
  python3 analyze_hd_sockets.py --d2h-multichip    results.csv
  python3 analyze_hd_sockets.py --h2d-throughput   results.csv
  python3 analyze_hd_sockets.py --h2d-latency      results.csv
  python3 analyze_hd_sockets.py --h2d-ping         ping_dir/
  python3 analyze_hd_sockets.py --h2d-multichip    results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CYCLES_PER_US = 1350.0


def human_bytes(n):
    for u in ("", "K", "M", "G"):
        if n < 1024:
            return f"{int(n)}{u}" if n == int(n) else f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}T"


# ── CSV loading ───────────────────────────────────────────────────────────────

# Arg names parsed from the gbench 'name' column.
# Keep in sync with ArgNames(...) in benchmark_hd_sockets.cpp.
_GBENCH_BENCH_ARGS = {
    "BM_D2HSocketThroughput": ["tray_id", "asic_location", "page_size", "fifo_size", "total_data"],
    "BM_H2DSocketThroughput": ["tray_id", "asic_location", "page_size", "fifo_size", "total_data", "mode_index"],
    "BM_D2HSocketLatency": ["tray_id", "asic_location", "page_size", "fifo_size"],
    "BM_H2DSocketLatency": ["tray_id", "asic_location", "page_size", "fifo_size", "mode_index"],
    "BM_D2HSocketPing": ["tray_id", "asic_location", "page_size", "fifo_size"],
    "BM_H2DSocketPing": ["tray_id", "asic_location", "page_size", "fifo_size", "mode_index"],
    "BM_D2HSocketMultiChipThroughput": ["chip_index", "fifo_size", "page_size", "total_data"],
    "BM_H2DSocketMultiChipThroughput": ["chip_index", "fifo_size", "page_size", "total_data"],
}

# Latency counters expected in the CSV.
_GBENCH_LATENCY_COUNTERS = [
    "num_iterations",
    "avg_us",
    "min_us",
    "max_us",
    "p50_us",
    "p99_us",
    "avg_cycles",
    "min_cycles",
    "max_cycles",
]


def load_gbench_csv(path):
    df = pd.read_csv(path)
    # Drop skipped/errored benchmarks (no timing data).
    if "error_message" in df.columns:
        df = df[df["error_message"].isna() | (df["error_message"] == "")].copy()
    return df


def _gbench_name_prefix(name: str) -> str:
    return name.split("/")[0]


def _gbench_extract_args(df: pd.DataFrame, arg_names: list) -> pd.DataFrame:
    """Extract benchmark arguments from the 'name' column.

    Handles both named (key:value) and positional formats, then
    renames 'fifo_size' -> 'socket_fifo_size' for script-wide compat.
    """

    def _parse(name):
        parts = name.split("/")[1:]  # skip benchmark name prefix
        if parts and ":" in parts[0]:
            # Named format: tray_id:1/asic_location:6/page_size:32768/...
            result = {}
            for part in parts:
                if ":" in part:
                    key, val = part.split(":", 1)
                    if key in arg_names:
                        try:
                            result[key] = int(val)
                        except ValueError:
                            # Ignore malformed integer values in benchmark name fields.
                            continue
            return result
        else:
            # Positional format (legacy): 32768/134217728/536870912
            return {k: int(parts[i]) for i, k in enumerate(arg_names) if i < len(parts)}

    result = df["name"].apply(_parse).apply(pd.Series).reset_index(drop=True)
    result.rename(columns={"fifo_size": "socket_fifo_size"}, inplace=True)
    return result


def _derive_throughput(df):
    """Compute per_page_us and throughput_gbps from real_time when counters are absent."""
    if "per_page_us" not in df.columns and "real_time" in df.columns:
        num_pages = df["total_data"] / df["page_size"]
        df["per_page_us"] = df["real_time"] * 1e6 / num_pages
    if "throughput_gbps" not in df.columns and "per_page_us" in df.columns:
        df["throughput_gbps"] = df["page_size"] / (df["per_page_us"] * 1e3)
    for col in ("per_page_us", "per_page_cycles", "throughput_gbps", "data_size", "num_iterations"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _drop_zero_throughput_rows(df, label):
    """Exclude invalid zero/negative throughput rows from analysis outputs."""
    if "throughput_gbps" not in df.columns:
        return df
    before = len(df)
    df = df[df["throughput_gbps"] > 0].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} {label} row(s) with non-positive throughput_gbps")
    if df.empty:
        raise ValueError(f"All {label} rows have non-positive throughput_gbps")
    return df


def load_gbench_d2h_throughput_csv(path):
    df = load_gbench_csv(path)
    df = df[df["name"].str.startswith("BM_D2HSocketThroughput")].copy()
    if df.empty:
        raise ValueError(f"No BM_D2HSocketThroughput rows found in {path}")

    arg_cols = _gbench_extract_args(df, _GBENCH_BENCH_ARGS["BM_D2HSocketThroughput"])
    df = pd.concat([df.reset_index(drop=True), arg_cols], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    if "label" in df.columns:
        df["device_coord"] = df["label"]

    _derive_throughput(df)

    required = ["page_size", "socket_fifo_size", "total_data", "per_page_us", "throughput_gbps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"gbench CSV is missing expected columns: {missing}")

    return _drop_zero_throughput_rows(df, "BM_D2HSocketThroughput")


_H2D_MODE_MAP = {0: "HOST_PUSH", 1: "DEVICE_PULL"}


def load_gbench_h2d_throughput_csv(path):
    df = load_gbench_csv(path)
    df = df[df["name"].str.startswith("BM_H2DSocketThroughput")].copy()
    if df.empty:
        raise ValueError(f"No BM_H2DSocketThroughput rows found in {path}")

    arg_cols = _gbench_extract_args(df, _GBENCH_BENCH_ARGS["BM_H2DSocketThroughput"])
    df = pd.concat([df.reset_index(drop=True), arg_cols], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    if "label" in df.columns:
        df["device_coord"] = df["label"]

    if "mode_index" in df.columns:
        df["h2d_mode"] = df["mode_index"].map(_H2D_MODE_MAP).fillna("UNKNOWN")
    elif "h2d_mode" not in df.columns:
        df["h2d_mode"] = "UNKNOWN"

    _derive_throughput(df)

    return _drop_zero_throughput_rows(df, "BM_H2DSocketThroughput")


def _load_gbench_latency(path, name_prefix):
    """Shared loader for latency and ping CSVs."""
    df = load_gbench_csv(path)
    df = df[df["name"].str.startswith(name_prefix)].copy()
    if df.empty:
        raise ValueError(f"No {name_prefix} rows found in {path}")

    arg_cols = _gbench_extract_args(df, _GBENCH_BENCH_ARGS[name_prefix])
    df = pd.concat([df.reset_index(drop=True), arg_cols], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    if "label" in df.columns:
        df["device_coord"] = df["label"]

    for col in _GBENCH_LATENCY_COUNTERS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Some benchmark CSV exports omit custom latency counters.
    # Fall back to real_time (seconds) converted to microseconds.
    if "num_iterations" not in df.columns and "iterations" in df.columns:
        df["num_iterations"] = pd.to_numeric(df["iterations"], errors="coerce")

    if "real_time" in df.columns:
        rt_us = pd.to_numeric(df["real_time"], errors="coerce") * 1e6
        for col in ("p50_us", "min_us", "max_us", "avg_us", "p99_us"):
            if col not in df.columns:
                df[col] = rt_us
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(rt_us)

        # Keep cycle fields consistent when not present.
        for col in ("avg_cycles", "min_cycles", "max_cycles"):
            if col not in df.columns:
                df[col] = df.get("avg_us", rt_us) * CYCLES_PER_US

    return df


def load_gbench_d2h_latency_csv(path):
    return _load_gbench_latency(path, "BM_D2HSocketLatency")


def load_gbench_d2h_ping_csv(path):
    return _load_gbench_latency(path, "BM_D2HSocketPing")


def _map_h2d_mode(df):
    if "mode_index" in df.columns:
        df["h2d_mode"] = df["mode_index"].map(_H2D_MODE_MAP).fillna("UNKNOWN")
    elif "h2d_mode" not in df.columns:
        df["h2d_mode"] = "UNKNOWN"
    return df


def load_gbench_h2d_latency_csv(path):
    return _map_h2d_mode(_load_gbench_latency(path, "BM_H2DSocketLatency"))


def load_gbench_h2d_ping_csv(path):
    return _map_h2d_mode(_load_gbench_latency(path, "BM_H2DSocketPing"))


def load_gbench_multichip_csv(path, name_prefix):
    df = load_gbench_csv(path)
    df = df[df["name"].str.startswith(name_prefix)].copy()
    if df.empty:
        raise ValueError(f"No {name_prefix} rows found in {path}")

    arg_cols = _gbench_extract_args(df, _GBENCH_BENCH_ARGS[name_prefix])
    df = pd.concat([df.reset_index(drop=True), arg_cols], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    _derive_throughput(df)

    for col in ("tray_id", "asic_location"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "label" in df.columns:
        df["mesh_coord"] = df["label"]

    df["chip_label"] = df.apply(
        lambda r: f"Tray{int(r.tray_id)}/ASIC{int(r.asic_location)} {r.get('mesh_coord', '')}",
        axis=1,
    )

    return _drop_zero_throughput_rows(df, name_prefix)


def load_gbench_d2h_multichip_csv(path):
    return load_gbench_multichip_csv(path, "BM_D2HSocketMultiChipThroughput")


def load_gbench_h2d_multichip_csv(path):
    return load_gbench_multichip_csv(path, "BM_H2DSocketMultiChipThroughput")


# benchmark prefix → (loader, runner)
_GBENCH_DISPATCH = {
    "BM_D2HSocketThroughput": ("load_gbench_d2h_throughput_csv", "run_d2h_throughput"),
    "BM_H2DSocketThroughput": ("load_gbench_h2d_throughput_csv", "run_h2d_throughput"),
    "BM_D2HSocketLatency": ("load_gbench_d2h_latency_csv", "run_d2h_latency"),
    "BM_D2HSocketPing": ("load_gbench_d2h_ping_csv", "run_d2h_latency"),
    "BM_H2DSocketLatency": ("load_gbench_h2d_latency_csv", "run_h2d_latency"),
    "BM_H2DSocketPing": ("load_gbench_h2d_ping_csv", "run_h2d_latency"),
    "BM_D2HSocketMultiChipThroughput": ("load_gbench_d2h_multichip_csv", "run_d2h_multichip"),
    "BM_H2DSocketMultiChipThroughput": ("load_gbench_h2d_multichip_csv", "run_h2d_multichip"),
}


def run_gbench(path, prefix=""):
    """Auto-detect benchmark type from CSV and dispatch to the right analysis."""
    df_raw = pd.read_csv(path)  # raw, unfiltered for detection
    if df_raw.empty or "name" not in df_raw.columns:
        raise ValueError(f"{path} does not look like a Google Benchmark CSV")

    prefix_counts = df_raw["name"].apply(_gbench_name_prefix).value_counts()
    detected = prefix_counts.index[0]

    print(f"  Detected benchmark: {detected}")

    if detected == "BM_D2HSocketThroughput":
        df = load_gbench_d2h_throughput_csv(path)
        d2h_tp_print_report(df)
        d2h_tp_plot(df, out=f"{prefix}d2h_throughput.png")
        d2h_tp_plot_mean(df, out=f"{prefix}d2h_throughput_mean.png")
        d2h_tp_plot_vs_fifo(df, out=f"{prefix}d2h_tp_vs_fifo.png")
        d2h_tp_plot_page_by_fifo_1g(df, out=f"{prefix}d2h_tp_page_by_fifo_1g.png")
        d2h_tp_plot_heatmap_grid(df, out=f"{prefix}d2h_tp_heatmap_grid.png")
        d2h_tp_export_csv(df, out=f"{prefix}d2h_throughput_summary.csv")
    elif detected in ("BM_D2HSocketLatency", "BM_D2HSocketPing"):
        loader = load_gbench_d2h_latency_csv if detected == "BM_D2HSocketLatency" else load_gbench_d2h_ping_csv
        df = loader(path)
        d2h_lat_print_report(df)
        d2h_lat_plot(df, out=f"{prefix}d2h_latency.png")
        d2h_lat_plot_breakdown(df, out=f"{prefix}d2h_latency_breakdown.png")
        _export_latency_csv(df, out=f"{prefix}d2h_latency_summary.csv")
    elif detected == "BM_H2DSocketThroughput":
        df = load_gbench_h2d_throughput_csv(path)
        h2d_tp_print_report(df)
        h2d_tp_plot(df, out=f"{prefix}h2d_throughput.png")
        h2d_tp_plot_mean(df, out=f"{prefix}h2d_throughput_mean.png")
        h2d_tp_plot_vs_fifo(df, out=f"{prefix}h2d_tp_vs_fifo.png")
        h2d_tp_plot_page_by_fifo_1g(df, out=f"{prefix}h2d_tp_page_by_fifo_1g.png")
        h2d_tp_plot_at_max_fifo(df, out=f"{prefix}h2d_tp_at_max_fifo.png")
        h2d_tp_export_csv(df, out=f"{prefix}h2d_throughput_summary.csv")
    elif detected in ("BM_H2DSocketLatency", "BM_H2DSocketPing"):
        loader = load_gbench_h2d_latency_csv if detected == "BM_H2DSocketLatency" else load_gbench_h2d_ping_csv
        df = loader(path)
        h2d_lat_print_report(df)
        h2d_lat_plot(df, out=f"{prefix}h2d_latency.png")
        h2d_lat_plot_breakdown(df, out=f"{prefix}h2d_latency_breakdown.png")
        _export_latency_csv(df, out=f"{prefix}h2d_latency_summary.csv", mode_col="h2d_mode")
    elif detected == "BM_D2HSocketMultiChipThroughput":
        df = load_gbench_d2h_multichip_csv(path)
        mc_print_report(df, direction="D2H")
        mc_plot_throughput_vs_fifo(df, out=f"{prefix}d2h_mc_throughput_vs_fifo.png", direction="D2H")
        mc_plot_bar_comparison(df, out=f"{prefix}d2h_mc_throughput_bar.png", direction="D2H")
        mc_plot_heatmap(df, out=f"{prefix}d2h_mc_throughput_heatmap.png", direction="D2H")
        mc_export_csv(df, out=f"{prefix}d2h_mc_throughput_summary.csv")
    elif detected == "BM_H2DSocketMultiChipThroughput":
        df = load_gbench_h2d_multichip_csv(path)
        mc_print_report(df, direction="H2D")
        mc_plot_throughput_vs_fifo(df, out=f"{prefix}h2d_mc_throughput_vs_fifo.png", direction="H2D")
        mc_plot_bar_comparison(df, out=f"{prefix}h2d_mc_throughput_bar.png", direction="H2D")
        mc_plot_heatmap(df, out=f"{prefix}h2d_mc_throughput_heatmap.png", direction="H2D")
        mc_export_csv(df, out=f"{prefix}h2d_mc_throughput_summary.csv")
    else:
        raise ValueError(
            f"No analysis registered for benchmark '{detected}'. " f"Known: {list(_GBENCH_DISPATCH.keys())}"
        )


# ── shared helpers ────────────────────────────────────────────────────────────


def _median_over_fifo(df, extra_groups=()):
    """Median throughput across FIFO sizes, grouped by page_size × total_data."""
    group_keys = list(extra_groups) + ["page_size", "total_data"]
    return (
        df.groupby(group_keys)
        .agg(
            throughput_gbps=("throughput_gbps", "median"),
            per_page_us=("per_page_us", "median"),
        )
        .reset_index()
    )


def _mean_over_fifo(df, extra_groups=()):
    """Mean throughput across FIFO sizes, grouped by page_size × total_data."""
    group_keys = list(extra_groups) + ["page_size", "total_data"]
    return (
        df.groupby(group_keys)
        .agg(
            throughput_gbps=("throughput_gbps", "mean"),
            per_page_us=("per_page_us", "mean"),
        )
        .reset_index()
    )


def _set_size_ticks(ax, values, rotation=45, fontsize=9):
    ax.set_xticks(values)
    ax.set_xticklabels([human_bytes(x) for x in values], rotation=rotation, fontsize=fontsize)
    ax.minorticks_off()


def _distinct_colors(n):
    # Build a palette sized exactly to the number of plotted series.
    cmap = plt.get_cmap("turbo")
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]


def _save_fig(fig, out, **tight_kw):
    fig.tight_layout(**tight_kw)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def _write_csv_tables(tables, out, float_format="%.6f"):
    with open(out, "w") as f:
        for i, t in enumerate(tables):
            if i > 0:
                f.write("\n")
            t.to_csv(f, float_format=float_format)
    print(f"  Saved {out}")


def _annotate_heatmap(ax, data, vmax, fontsize=5):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="white" if v > vmax * 0.7 else "black",
                )


def _export_latency_csv(df, out, mode_col=None):
    fifos = sorted(df.socket_fifo_size.unique())
    metric_cols = ["p50_us", "min_us", "max_us", "p99_us", "avg_us"]
    modes = sorted(df[mode_col].unique()) if mode_col else [None]

    tables = []
    for mode in modes:
        sub = df[df[mode_col] == mode] if mode else df
        label = f"page_size ({mode})" if mode else "page_size"
        result = pd.DataFrame(index=sorted(sub.page_size.unique()))
        result.index.name = label
        for fs in fifos:
            g = sub[sub.socket_fifo_size == fs].set_index("page_size")
            tag = human_bytes(fs)
            for col in metric_cols:
                result[f"{col} (FIFO={tag})"] = g[col]
        tables.append(result)

    _write_csv_tables(tables, out, float_format="%.4f")


def _resolve_input_path(path):
    return Path(path).expanduser().resolve()


def _latency_iterations(df):
    if "num_iterations" in df.columns:
        return int(pd.to_numeric(df["num_iterations"], errors="coerce").dropna().iloc[0])
    if "iterations" in df.columns:
        return int(pd.to_numeric(df["iterations"], errors="coerce").dropna().iloc[0])
    return 0


def _auto_chip_prefix(csv_path, explicit_prefix):
    if explicit_prefix:
        return explicit_prefix
    stem = Path(csv_path).stem.lower()
    if "_x1" in stem or stem.endswith("x1"):
        return "x1_"
    if "_x8" in stem or stem.endswith("x8"):
        return "x8_"
    return explicit_prefix


# ══════════════════════════════════════════════════════════════════════════════
#  D2H THROUGHPUT
# ══════════════════════════════════════════════════════════════════════════════


def d2h_tp_print_report(df):
    agg = _median_over_fifo(df)
    nf = df.socket_fifo_size.nunique()

    print(f"\n{'='*70}\n  D2H Steady-State Throughput  ({len(df)} rows)\n{'='*70}")
    print(f"  Page sizes : {[human_bytes(x) for x in sorted(df.page_size.unique())]}")
    print(
        f"  FIFO sizes : {nf} values, {human_bytes(df.socket_fifo_size.min())} .. "
        f"{human_bytes(df.socket_fifo_size.max())}"
    )
    print(f"  Total data : {[human_bytes(x) for x in sorted(df.total_data.unique())]}")

    pk = agg.sort_values("throughput_gbps", ascending=False).iloc[0]
    print(
        f"\n  Peak: {pk.throughput_gbps:.3f} GB/s "
        f"(page={human_bytes(pk.page_size)}, total={human_bytes(pk.total_data)})"
    )

    tds = sorted(agg.total_data.unique())
    hdr = f"  {'page':>6}" + "".join(f"{human_bytes(t):>8}" for t in tds)
    print(f"\n  Throughput (GB/s) -- median across {nf} FIFO sizes:\n{hdr}\n  {'-'*(len(hdr)-2)}")
    for ps in sorted(agg.page_size.unique()):
        row = f"  {human_bytes(ps):>6}"
        for td in tds:
            c = agg[(agg.page_size == ps) & (agg.total_data == td)]
            row += f"{c.throughput_gbps.values[0]:>8.3f}" if len(c) else f"{'--':>8}"
        print(row)

    td = df.total_data.max()
    sub = df[df.total_data == td]
    print(f"\n  FIFO impact at total_data={human_bytes(td)}:")
    print(f"  {'page':>6} {'min':>8} {'median':>8} {'max':>8} {'CV%':>6}")
    print(f"  {'-'*38}")
    for ps in sorted(sub.page_size.unique()):
        g = sub[sub.page_size == ps].throughput_gbps
        cv = g.std() / g.mean() * 100 if g.mean() > 0 else 0
        print(f"  {human_bytes(ps):>6} {g.min():>8.3f} {g.median():>8.3f} {g.max():>8.3f} {cv:>5.1f}%")

    agg2 = _median_over_fifo(df[df.total_data == td])
    print(f"\n  Per-page timing (total_data={human_bytes(td)}):")
    print(f"  {'page':>6} {'us/page':>10} {'cycles':>10} {'GB/s':>8}\n  {'-'*38}")
    for _, r in agg2.sort_values("page_size").iterrows():
        print(
            f"  {human_bytes(r.page_size):>6} {r.per_page_us:>10.3f} "
            f"{r.per_page_us * CYCLES_PER_US:>10.0f} {r.throughput_gbps:>8.3f}"
        )


def d2h_tp_plot(df, out="d2h_throughput.png"):
    fs_max = df.socket_fifo_size.max()
    sub = df[df.socket_fifo_size == fs_max]
    fig, ax = plt.subplots(figsize=(12, 6))
    for td in sorted(sub.total_data.unique()):
        g = sub[sub.total_data == td].sort_values("page_size")
        ax.plot(g.page_size, g.throughput_gbps, "o-", label=f"{human_bytes(td)}", ms=5, lw=2)
    ax.set(
        xscale="log",
        xlabel="Page Size",
        ylabel="Throughput (GB/s)",
        title=f"D2H Throughput vs Page Size — FIFO={human_bytes(fs_max)} (max), each line = one total transfer size",
    )
    _set_size_ticks(ax, sorted(sub.page_size.unique()))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, title="Total Data")
    _save_fig(fig, out)


def d2h_tp_plot_mean(df, out="d2h_throughput_mean.png"):
    td_max = df.total_data.max()
    sub = df[df.total_data == td_max]
    agg = (
        sub.groupby("page_size")
        .agg(
            fifo_min=("throughput_gbps", "min"),
            fifo_median=("throughput_gbps", "median"),
            fifo_mean=("throughput_gbps", "mean"),
            fifo_max=("throughput_gbps", "max"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = agg.page_size.values
    y_min = agg.fifo_min.values
    y_med = agg.fifo_median.values
    y_mean = agg.fifo_mean.values
    y_max = agg.fifo_max.values

    # Show spread first so center trends stay readable.
    ax.fill_between(x, y_min, y_max, color="#90CAF9", alpha=0.22, label="min-max band")
    ax.plot(x, y_min, color="#546E7A", ls=":", marker="v", ms=4, lw=1.6, label="min")
    ax.plot(x, y_max, color="#37474F", ls=":", marker="^", ms=4, lw=1.6, label="max")
    ax.plot(x, y_med, color="#512DA8", ls="-", marker="o", ms=5, lw=2.6, label="median")
    ax.plot(x, y_mean, color="#F57C00", ls="--", marker="s", ms=5, lw=2.2, label="mean")
    ax.set(
        xscale="log",
        xlabel="Page Size",
        ylabel="Throughput (GB/s)",
        title=f"D2H Throughput vs Page Size — total={human_bytes(td_max)}, min/median/mean/max across FIFO sizes",
    )
    _set_size_ticks(ax, sorted(agg.page_size.unique()))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9, title="FIFO statistic")
    _save_fig(fig, out)


def d2h_tp_plot_vs_fifo(df, out="d2h_tp_vs_fifo.png"):
    td = df.total_data.max()
    sub = df[df.total_data == td]
    fig, ax = plt.subplots(figsize=(12, 6))
    pages = sorted(sub.page_size.unique())
    colors = _distinct_colors(len(pages))
    for idx, ps in enumerate(pages):
        g = sub[sub.page_size == ps].sort_values("socket_fifo_size")
        ax.plot(g.socket_fifo_size, g.throughput_gbps, "o-", color=colors[idx], label=human_bytes(ps), ms=4, lw=1.5)
    ax.set(
        xscale="log",
        xlabel="Socket FIFO Size",
        ylabel="GB/s",
        title=f"D2H Throughput vs Socket FIFO Size — total={human_bytes(td)}, each line = one page size",
    )
    _set_size_ticks(ax, sorted(sub.socket_fifo_size.unique()), fontsize=7)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, title="Page Size")
    _save_fig(fig, out)


def d2h_tp_plot_page_by_fifo_1g(df, out="d2h_tp_page_by_fifo_1g.png"):
    td_target = (1 << 30) if (1 << 30) in set(df.total_data.unique()) else df.total_data.max()
    sub = df[df.total_data == td_target]
    fig, ax = plt.subplots(figsize=(12, 6))
    fifos = sorted(sub.socket_fifo_size.unique())
    colors = _distinct_colors(len(fifos))
    for idx, fs in enumerate(fifos):
        g = sub[sub.socket_fifo_size == fs].sort_values("page_size")
        ax.plot(g.page_size, g.throughput_gbps, "o-", color=colors[idx], label=human_bytes(fs), ms=4, lw=1.5)
    ax.set(
        xscale="log",
        xlabel="Page Size",
        ylabel="GB/s",
        title=f"D2H Throughput vs Page Size — total={human_bytes(td_target)}, each line = one FIFO size",
    )
    _set_size_ticks(ax, sorted(sub.page_size.unique()), fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, title="FIFO Size")
    _save_fig(fig, out)


def d2h_tp_plot_heatmap_grid(df, out="d2h_tp_heatmap_grid.png"):
    td_vals = sorted(df.total_data.unique())
    ncols, n = 4, len(td_vals)
    nrows = -(-n // ncols)
    vmin, vmax = df.throughput_gbps.min(), df.throughput_gbps.max()
    all_ps, all_fs = sorted(df.page_size.unique()), sorted(df.socket_fifo_size.unique())

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.5 * nrows))
    for idx, ax in enumerate(axes.flatten()):
        if idx >= n:
            ax.axis("off")
            continue
        td = td_vals[idx]
        pv = (
            df[df.total_data == td]
            .pivot_table(index="page_size", columns="socket_fifo_size", values="throughput_gbps")
            .reindex(index=all_ps, columns=all_fs)
        )
        im = ax.imshow(pv.values, aspect="auto", cmap="YlGn", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"total_data = {human_bytes(td)}", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(all_fs)))
        ax.set_xticklabels([human_bytes(f) for f in all_fs], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(all_ps)))
        ax.set_yticklabels([human_bytes(p) for p in all_ps], fontsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("Page Size", fontsize=9)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("FIFO Size", fontsize=9)
        _annotate_heatmap(ax, pv.values, vmax, fontsize=4)

    fig.suptitle(
        "D2H Throughput (GB/s) — each subplot = one total transfer size, X = FIFO size, Y = page size",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    cb = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cb, label="GB/s")
    fig.subplots_adjust(left=0.04, right=0.9, hspace=0.45, wspace=0.25)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")  # no tight_layout: uses subplots_adjust instead


def d2h_tp_export_csv(df, out="d2h_throughput_summary.csv"):
    repr_fifos = [f for f in [1024, 4096, 32768, 512 * 1024 * 1024] if f in df.socket_fifo_size.values]
    nf = df.socket_fifo_size.nunique()

    def pivot(sub, label):
        pv = sub.pivot_table(index="page_size", columns="total_data", values="throughput_gbps")
        pv.columns = [f"D0 Read {human_bytes(c)}" for c in pv.columns]
        pv.index.name = label
        return pv

    tables = [pivot(_median_over_fifo(df), f"page_size (median over {nf} FIFO sizes)")]
    for fs in repr_fifos:
        sub = df[df.socket_fifo_size == fs]
        if not sub.empty:
            tables.append(pivot(sub, f"page_size (FIFO={human_bytes(fs)})"))
    _write_csv_tables(tables, out)


def run_d2h_throughput(path, prefix=""):
    df = load_gbench_d2h_throughput_csv(path)
    d2h_tp_print_report(df)
    d2h_tp_plot(df, out=f"{prefix}d2h_throughput.png")
    d2h_tp_plot_mean(df, out=f"{prefix}d2h_throughput_mean.png")
    d2h_tp_plot_vs_fifo(df, out=f"{prefix}d2h_tp_vs_fifo.png")
    d2h_tp_plot_page_by_fifo_1g(df, out=f"{prefix}d2h_tp_page_by_fifo_1g.png")
    d2h_tp_plot_heatmap_grid(df, out=f"{prefix}d2h_tp_heatmap_grid.png")
    d2h_tp_export_csv(df, out=f"{prefix}d2h_throughput_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  D2H LATENCY
# ══════════════════════════════════════════════════════════════════════════════


def d2h_lat_print_report(df):
    print(f"\n{'='*70}\n  D2H Round-Trip Latency  ({len(df)} rows)\n{'='*70}")
    print(f"  Page sizes : {[human_bytes(x) for x in sorted(df.page_size.unique())]}")
    print(f"  FIFO sizes : {[human_bytes(x) for x in sorted(df.socket_fifo_size.unique())]}")
    print(f"  Iterations : {_latency_iterations(df)}")

    fifos = sorted(df.socket_fifo_size.unique())
    hdr = f"  {'page':>6}" + "".join(f"{human_bytes(f):>10}" for f in fifos)
    print(f"\n  p50 latency (us):\n{hdr}\n  {'-'*(len(hdr)-2)}")
    for ps in sorted(df.page_size.unique()):
        row = f"  {human_bytes(ps):>6}"
        for fs in fifos:
            c = df[(df.page_size == ps) & (df.socket_fifo_size == fs)]
            row += f"{c.p50_us.values[0]:>10.2f}" if len(c) else f"{'--':>10}"
        print(row)

    fs_max = df.socket_fifo_size.max()
    sub = df[df.socket_fifo_size == fs_max]
    print(f"\n  Latency spread at FIFO={human_bytes(fs_max)}:")
    print(f"  {'page':>6} {'min':>8} {'p50':>8} {'p99':>8} {'max':>8} {'avg':>8}")
    print(f"  {'-'*50}")
    for ps in sorted(sub.page_size.unique()):
        r = sub[sub.page_size == ps].iloc[0]
        print(
            f"  {human_bytes(ps):>6} {r.min_us:>8.2f} {r.p50_us:>8.2f} "
            f"{r.p99_us:>8.2f} {r.max_us:>8.2f} {r.avg_us:>8.2f}"
        )


def d2h_lat_plot(df, out="d2h_latency.png"):
    agg = (
        df.groupby("page_size")
        .agg(
            fifo_min=("p50_us", "min"),
            fifo_median=("p50_us", "median"),
            fifo_max=("p50_us", "max"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(agg.page_size, agg.fifo_min, "o:", color="#546E7A", ms=4, lw=1.6, label="min")
    ax.plot(agg.page_size, agg.fifo_median, "o-", color="#512DA8", ms=5, lw=2.4, label="median")
    ax.plot(agg.page_size, agg.fifo_max, "o:", color="#37474F", ms=4, lw=1.6, label="max")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Page Size (bytes)",
        ylabel="Latency (us)",
        title="D2H Round-Trip Latency (min/median/max across FIFO sizes)",
    )
    _set_size_ticks(ax, sorted(agg.page_size.unique()))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, title="FIFO statistic")
    _save_fig(fig, out)


def d2h_lat_plot_breakdown(df, out="d2h_latency_breakdown.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pages = sorted(df.page_size.unique())
    agg = (
        df.groupby("page_size")
        .agg(
            p50_min=("p50_us", "min"),
            p50_median=("p50_us", "median"),
            p50_max=("p50_us", "max"),
            max_min=("max_us", "min"),
            max_median=("max_us", "median"),
            max_max=("max_us", "max"),
        )
        .reindex(pages)
        .reset_index()
    )

    for ax, ylabel, title in zip(
        axes,
        ["p50 Latency (us)", "Max Latency (us)"],
        ["D2H p50 Latency (across FIFO)", "D2H Max Latency (across FIFO)"],
    ):
        if "p50" in title:
            ax.plot(agg.page_size, agg.p50_min, "o:", color="#546E7A", ms=4, lw=1.6, label="min")
            ax.plot(agg.page_size, agg.p50_median, "o-", color="#512DA8", ms=5, lw=2.4, label="median")
            ax.plot(agg.page_size, agg.p50_max, "o:", color="#37474F", ms=4, lw=1.6, label="max")
        else:
            ax.plot(agg.page_size, agg.max_min, "o:", color="#546E7A", ms=4, lw=1.6, label="min")
            ax.plot(agg.page_size, agg.max_median, "o-", color="#512DA8", ms=5, lw=2.4, label="median")
            ax.plot(agg.page_size, agg.max_max, "o:", color="#37474F", ms=4, lw=1.6, label="max")
        ax.set(xscale="log", xlabel="Page Size", ylabel=ylabel, title=title)
        _set_size_ticks(ax, pages, fontsize=8)
        ax.legend(fontsize=8, title="FIFO statistic")
        ax.grid(alpha=0.2, axis="y")

    _save_fig(fig, out)


def run_d2h_latency(path, prefix=""):
    df = load_gbench_d2h_latency_csv(path)
    d2h_lat_print_report(df)
    d2h_lat_plot(df, out=f"{prefix}d2h_latency.png")
    d2h_lat_plot_breakdown(df, out=f"{prefix}d2h_latency_breakdown.png")
    _export_latency_csv(df, out=f"{prefix}d2h_latency_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  H2D THROUGHPUT
# ══════════════════════════════════════════════════════════════════════════════


def h2d_tp_print_report(df):
    modes = sorted(df.h2d_mode.unique())
    nf = df.socket_fifo_size.nunique()
    agg = _median_over_fifo(df, extra_groups=["h2d_mode"])

    print(f"\n{'='*70}\n  H2D Steady-State Throughput  ({len(df)} rows)\n{'='*70}")
    print(f"  Modes      : {modes}")
    print(f"  Page sizes : {[human_bytes(x) for x in sorted(df.page_size.unique())]}")
    print(
        f"  FIFO sizes : {nf} values, {human_bytes(df.socket_fifo_size.min())} .. "
        f"{human_bytes(df.socket_fifo_size.max())}"
    )
    print(f"  Total data : {[human_bytes(x) for x in sorted(df.total_data.unique())]}")

    for mode in modes:
        sub = df[df.h2d_mode == mode]
        pk = sub.sort_values("throughput_gbps", ascending=False).iloc[0]
        print(
            f"\n  Peak {mode}: {pk.throughput_gbps:.3f} GB/s "
            f"(page={human_bytes(pk.page_size)}, FIFO={human_bytes(pk.socket_fifo_size)}, "
            f"total={human_bytes(pk.total_data)})"
        )

    tds = sorted(agg.total_data.unique())
    for mode in modes:
        m_agg = agg[agg.h2d_mode == mode]
        hdr = f"  {'page':>6}" + "".join(f"{human_bytes(t):>8}" for t in tds)
        print(f"\n  {mode} throughput (GB/s) -- median across {nf} FIFO sizes:\n{hdr}\n  {'-'*(len(hdr)-2)}")
        for ps in sorted(m_agg.page_size.unique()):
            row = f"  {human_bytes(ps):>6}"
            for td in tds:
                c = m_agg[(m_agg.page_size == ps) & (m_agg.total_data == td)]
                row += f"{c.throughput_gbps.values[0]:>8.3f}" if len(c) else f"{'--':>8}"
            print(row)

    td = df.total_data.max()
    for mode in modes:
        sub = df[(df.total_data == td) & (df.h2d_mode == mode)]
        if sub.empty:
            continue
        print(f"\n  {mode} FIFO impact at total_data={human_bytes(td)}:")
        print(f"  {'page':>6} {'min':>8} {'median':>8} {'max':>8} {'CV%':>6}")
        print(f"  {'-'*38}")
        for ps in sorted(sub.page_size.unique()):
            g = sub[sub.page_size == ps].throughput_gbps
            cv = g.std() / g.mean() * 100 if g.mean() > 0 else 0
            print(f"  {human_bytes(ps):>6} {g.min():>8.3f} {g.median():>8.3f} {g.max():>8.3f} {cv:>5.1f}%")

    agg2 = _median_over_fifo(df[df.total_data == td], extra_groups=["h2d_mode"])
    print(f"\n  Per-page timing (total_data={human_bytes(td)}, median over FIFO sizes):")
    print(f"  {'page':>6} {'mode':>14} {'us/page':>10} {'cycles':>10} {'GB/s':>8}\n  {'-'*52}")
    for ps in sorted(agg2.page_size.unique()):
        for mode in modes:
            r = agg2[(agg2.page_size == ps) & (agg2.h2d_mode == mode)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(
                f"  {human_bytes(r.page_size):>6} {mode:>14} {r.per_page_us:>10.3f} "
                f"{r.per_page_us * CYCLES_PER_US:>10.0f} {r.throughput_gbps:>8.3f}"
            )


def h2d_tp_plot(df, out="h2d_throughput.png"):
    td_max = df.total_data.max()
    fs_max = df.socket_fifo_size.max()
    sub = df[(df.total_data == td_max) & (df.socket_fifo_size == fs_max)]
    modes = sorted(sub.h2d_mode.unique())
    agg = sub.groupby(["h2d_mode", "page_size"]).agg(throughput_gbps=("throughput_gbps", "median")).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    for mode in modes:
        g = agg[agg.h2d_mode == mode].sort_values("page_size")
        ax.plot(g.page_size, g.throughput_gbps, "o-", label=mode, ms=5, lw=2)
    ax.set(
        xscale="log",
        xlabel="Page Size",
        ylabel="GB/s",
        title=f"H2D Throughput vs Page Size — FIFO={human_bytes(fs_max)} (max), total={human_bytes(td_max)} (max)",
    )
    _set_size_ticks(ax, sorted(agg.page_size.unique()))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=10, title="H2D Mode")
    _save_fig(fig, out)


def h2d_tp_plot_mean(df, out="h2d_throughput_mean.png"):
    td_max = df.total_data.max()
    sub = df[df.total_data == td_max]
    modes = sorted(sub.h2d_mode.unique())

    fig, axes = plt.subplots(1, len(modes), figsize=(8 * len(modes), 6), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        mode_df = sub[sub.h2d_mode == mode]
        g = (
            mode_df.groupby("page_size")
            .agg(
                fifo_min=("throughput_gbps", "min"),
                fifo_median=("throughput_gbps", "median"),
                fifo_mean=("throughput_gbps", "mean"),
                fifo_max=("throughput_gbps", "max"),
            )
            .reset_index()
        )
        x = g.page_size.values
        y_min = g.fifo_min.values
        y_med = g.fifo_median.values
        y_mean = g.fifo_mean.values
        y_max = g.fifo_max.values

        ax.fill_between(x, y_min, y_max, color="#90CAF9", alpha=0.22, label="min-max band")
        ax.plot(x, y_min, color="#546E7A", ls=":", marker="v", ms=4, lw=1.6, label="min")
        ax.plot(x, y_max, color="#37474F", ls=":", marker="^", ms=4, lw=1.6, label="max")
        ax.plot(x, y_med, color="#512DA8", ls="-", marker="o", ms=5, lw=2.6, label="median")
        ax.plot(x, y_mean, color="#F57C00", ls="--", marker="s", ms=5, lw=2.2, label="mean")
        ax.set(
            xscale="log",
            xlabel="Page Size",
            ylabel="GB/s" if idx == 0 else "",
            title=f"H2D mode: {mode}",
        )
        _set_size_ticks(ax, sorted(g.page_size.unique()), fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, title="FIFO statistic")

    fig.suptitle(
        f"H2D Throughput vs Page Size — total={human_bytes(td_max)}, min/median/mean/max across FIFO sizes",
        fontsize=13,
        fontweight="bold",
    )
    _save_fig(fig, out, rect=[0, 0, 1, 0.96])


def h2d_tp_plot_vs_fifo(df, out="h2d_tp_vs_fifo.png"):
    td = df.total_data.max()
    sub = df[df.total_data == td]
    modes = sorted(sub.h2d_mode.unique())
    all_page_sizes = sorted(sub.page_size.unique())
    rep_pages = [p for p in [64, 256, 1024, 4096, 16384, 32768, 65536, 131072, 262144] if p in all_page_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        mode_df = sub[sub.h2d_mode == mode]
        colors = _distinct_colors(len(rep_pages))
        for color_idx, ps in enumerate(rep_pages):
            g = mode_df[mode_df.page_size == ps].sort_values("socket_fifo_size")
            if not g.empty:
                ax.plot(
                    g.socket_fifo_size,
                    g.throughput_gbps,
                    "o-",
                    color=colors[color_idx],
                    label=human_bytes(ps),
                    ms=4,
                    lw=1.5,
                )
        fifos = sorted(mode_df.socket_fifo_size.unique())
        tick_fifos = fifos[:: max(1, len(fifos) // 6)]
        ax.set(
            xscale="log",
            xlabel="Socket FIFO Size",
            ylabel="Throughput (GB/s)" if idx == 0 else "",
            title=f"H2D mode: {mode}",
        )
        _set_size_ticks(ax, tick_fifos, fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9, title="Page Size")

    fig.suptitle(
        f"H2D Throughput vs Socket FIFO Size — total={human_bytes(td)} (max), each line = one page size",
        fontsize=13,
        fontweight="bold",
    )
    _save_fig(fig, out, rect=[0, 0, 1, 0.96])


def h2d_tp_plot_page_by_fifo_1g(df, out="h2d_tp_page_by_fifo_1g.png"):
    td_target = (1 << 30) if (1 << 30) in set(df.total_data.unique()) else df.total_data.max()
    sub = df[df.total_data == td_target]
    modes = sorted(sub.h2d_mode.unique())

    fig, axes = plt.subplots(1, len(modes), figsize=(8 * len(modes), 6), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        mode_df = sub[sub.h2d_mode == mode]
        fifos = sorted(mode_df.socket_fifo_size.unique())
        colors = _distinct_colors(len(fifos))
        for color_idx, fs in enumerate(fifos):
            g = mode_df[mode_df.socket_fifo_size == fs].sort_values("page_size")
            if not g.empty:
                ax.plot(
                    g.page_size,
                    g.throughput_gbps,
                    "o-",
                    color=colors[color_idx],
                    label=human_bytes(fs),
                    ms=4,
                    lw=1.5,
                )
        ax.set(
            xscale="log",
            xlabel="Page Size",
            ylabel="Throughput (GB/s)" if idx == 0 else "",
            title=f"H2D mode: {mode}",
        )
        _set_size_ticks(ax, sorted(mode_df.page_size.unique()), fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9, title="FIFO Size")

    fig.suptitle(
        f"H2D Throughput vs Page Size — total={human_bytes(td_target)}, each line = one FIFO size",
        fontsize=13,
        fontweight="bold",
    )
    _save_fig(fig, out, rect=[0, 0, 1, 0.96])


def h2d_tp_plot_at_max_fifo(df, out="h2d_tp_at_max_fifo.png"):
    fs_max = df.socket_fifo_size.max()
    sub = df[df.socket_fifo_size == fs_max]
    modes = sorted(sub.h2d_mode.unique())
    td_vals = sorted(sub.total_data.unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        mode_df = sub[sub.h2d_mode == mode]
        for td in td_vals:
            g = mode_df[mode_df.total_data == td].sort_values("page_size")
            if not g.empty:
                ax.plot(g.page_size, g.throughput_gbps, "o-", label=human_bytes(td), ms=5, lw=2)
        ax.set(
            xscale="log",
            xlabel="Page Size",
            ylabel="GB/s" if idx == 0 else "",
            title=f"H2D mode: {mode}  — FIFO={human_bytes(fs_max)} (max)",
        )
        _set_size_ticks(ax, sorted(mode_df.page_size.unique()), fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9, title="Total Data")

    fig.suptitle(
        f"H2D Throughput vs Page Size — FIFO={human_bytes(fs_max)} (max), each line = one total transfer size",
        fontsize=13,
        fontweight="bold",
    )
    _save_fig(fig, out, rect=[0, 0, 1, 0.96])


def h2d_tp_export_csv(df, out="h2d_throughput_summary.csv"):
    modes = sorted(df.h2d_mode.unique())
    nf = df.socket_fifo_size.nunique()
    repr_fifos = [f for f in [1024, 16384, 524288, 1048576] if f in df.socket_fifo_size.values]

    def pivot_combined(sub, label):
        parts = []
        for mode in modes:
            pv = sub[sub.h2d_mode == mode].pivot_table(
                index="page_size", columns="total_data", values="throughput_gbps"
            )
            pv.columns = [f"{mode} {human_bytes(c)}" for c in pv.columns]
            parts.append(pv)
        merged = parts[0].join(parts[1:], how="outer")
        merged.index.name = label
        return merged

    agg = _median_over_fifo(df, extra_groups=["h2d_mode"])
    tables = [pivot_combined(agg, f"page_size (median over {nf} FIFO sizes)")]
    for fs in repr_fifos:
        sub = df[df.socket_fifo_size == fs]
        if not sub.empty:
            tables.append(pivot_combined(sub, f"page_size (FIFO={human_bytes(fs)})"))
    _write_csv_tables(tables, out)


def run_h2d_throughput(path, prefix=""):
    df = load_gbench_h2d_throughput_csv(path)
    h2d_tp_print_report(df)
    h2d_tp_plot(df, out=f"{prefix}h2d_throughput.png")
    h2d_tp_plot_mean(df, out=f"{prefix}h2d_throughput_mean.png")
    h2d_tp_plot_vs_fifo(df, out=f"{prefix}h2d_tp_vs_fifo.png")
    h2d_tp_plot_page_by_fifo_1g(df, out=f"{prefix}h2d_tp_page_by_fifo_1g.png")
    h2d_tp_plot_at_max_fifo(df, out=f"{prefix}h2d_tp_at_max_fifo.png")
    h2d_tp_export_csv(df, out=f"{prefix}h2d_throughput_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  H2D LATENCY
# ══════════════════════════════════════════════════════════════════════════════


def h2d_lat_print_report(df):
    modes = sorted(df.h2d_mode.unique())
    print(f"\n{'='*70}\n  H2D Round-Trip Latency  ({len(df)} rows)\n{'='*70}")
    print(f"  Modes      : {modes}")
    print(f"  Page sizes : {[human_bytes(x) for x in sorted(df.page_size.unique())]}")
    print(f"  FIFO sizes : {[human_bytes(x) for x in sorted(df.socket_fifo_size.unique())]}")
    print(f"  Iterations : {_latency_iterations(df)}")

    fs_max = df.socket_fifo_size.max()
    print(f"\n  p50 latency (us) at FIFO={human_bytes(fs_max)}:")
    print(f"  {'page':>6}", end="")
    for m in modes:
        print(f" {m:>12}", end="")
    print(f" {'delta':>10}")
    print(f"  {'-'*50}")
    for ps in sorted(df.page_size.unique()):
        print(f"  {human_bytes(ps):>6}", end="")
        vals = {}
        for m in modes:
            c = df[(df.page_size == ps) & (df.socket_fifo_size == fs_max) & (df.h2d_mode == m)]
            val = c.p50_us.values[0] if len(c) else None
            vals[m] = val
            print(f" {val:>12.2f}" if val else f" {'--':>12}", end="")
        if len(vals) == 2 and all(v is not None for v in vals.values()):
            delta = list(vals.values())[1] - list(vals.values())[0]
            print(f" {delta:>+10.2f}")
        else:
            print()

    for mode in modes:
        sub = df[df.h2d_mode == mode]
        fifos = sorted(sub.socket_fifo_size.unique())
        print(f"\n  {mode} - p50 latency (us):")
        hdr = f"  {'page':>6}" + "".join(f"{human_bytes(f):>10}" for f in fifos)
        print(f"{hdr}\n  {'-'*(len(hdr)-2)}")
        for ps in sorted(sub.page_size.unique()):
            row = f"  {human_bytes(ps):>6}"
            for fs in fifos:
                c = sub[(sub.page_size == ps) & (sub.socket_fifo_size == fs)]
                row += f"{c.p50_us.values[0]:>10.2f}" if len(c) else f"{'--':>10}"
            print(row)


def h2d_lat_plot(df, out="h2d_latency.png"):
    modes = sorted(df.h2d_mode.unique())

    fig, axes = plt.subplots(1, len(modes), figsize=(8 * len(modes), 6), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        mode_df = df[df.h2d_mode == mode]
        g = (
            mode_df.groupby("page_size")
            .agg(
                fifo_min=("p50_us", "min"),
                fifo_median=("p50_us", "median"),
                fifo_max=("p50_us", "max"),
            )
            .reset_index()
        )
        ax.plot(g.page_size, g.fifo_min, "o:", color="#546E7A", ms=4, lw=1.6, label="min")
        ax.plot(g.page_size, g.fifo_median, "o-", color="#512DA8", ms=5, lw=2.4, label="median")
        ax.plot(g.page_size, g.fifo_max, "o:", color="#37474F", ms=4, lw=1.6, label="max")
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Page Size (bytes)",
            ylabel="Latency (us)" if idx == 0 else "",
            title=f"H2D mode: {mode}",
        )
        _set_size_ticks(ax, sorted(g.page_size.unique()))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, title="FIFO statistic")

    fig.suptitle("H2D Round-Trip Latency (min/median/max across FIFO sizes)", fontsize=13, fontweight="bold")
    _save_fig(fig, out, rect=[0, 0, 1, 0.96])


def h2d_lat_plot_breakdown(df, out="h2d_latency_breakdown.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    modes = sorted(df.h2d_mode.unique())

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        sub = df[df.h2d_mode == mode]
        g = (
            sub.groupby("page_size")
            .agg(
                fifo_min=("p50_us", "min"),
                fifo_median=("p50_us", "median"),
                fifo_max=("p50_us", "max"),
            )
            .reset_index()
        )
        ax.plot(g.page_size, g.fifo_min, "o:", color="#546E7A", ms=4, lw=1.6, label="min")
        ax.plot(g.page_size, g.fifo_median, "o-", color="#512DA8", ms=5, lw=2.4, label="median")
        ax.plot(g.page_size, g.fifo_max, "o:", color="#37474F", ms=4, lw=1.6, label="max")
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Page Size (bytes)",
            ylabel="Latency (us)" if idx == 0 else "",
            title=f"{mode}",
        )
        _set_size_ticks(ax, sorted(g.page_size.unique()))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best", title="FIFO statistic")

    fig.suptitle("H2D Latency (min/median/max across FIFO sizes)", fontsize=13, fontweight="bold")
    _save_fig(fig, out, rect=[0, 0, 1, 0.96])


def h2d_lat_plot_fifo_impact(df, out="h2d_latency_vs_fifo.png"):
    modes = sorted(df.h2d_mode.unique())
    page_sizes = [p for p in [64, 256, 1024, 4096, 16384] if p in df.page_size.values]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        sub = df[df.h2d_mode == mode]
        for ps in page_sizes:
            g = sub[sub.page_size == ps].sort_values("socket_fifo_size")
            if not g.empty:
                ax.plot(g.socket_fifo_size, g.p50_us, "o-", label=human_bytes(ps), ms=5, lw=2)
        ax.set(
            xscale="log",
            xlabel="Socket FIFO Size (bytes)",
            ylabel="Latency (us)" if idx == 0 else "",
            title=f"{mode}",
        )
        _set_size_ticks(ax, sorted(sub.socket_fifo_size.unique()), fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, title="Page Size")

    fig.suptitle("H2D Latency vs FIFO Size", fontsize=13, fontweight="bold")
    _save_fig(fig, out, rect=[0, 0, 1, 0.96])


def run_h2d_latency(path, prefix=""):
    df = load_gbench_h2d_latency_csv(path)
    h2d_lat_print_report(df)
    h2d_lat_plot(df, out=f"{prefix}h2d_latency.png")
    h2d_lat_plot_breakdown(df, out=f"{prefix}h2d_latency_breakdown.png")
    h2d_lat_plot_fifo_impact(df, out=f"{prefix}h2d_latency_vs_fifo.png")
    _export_latency_csv(df, out=f"{prefix}h2d_latency_summary.csv", mode_col="h2d_mode")


# ══════════════════════════════════════════════════════════════════════════════
#  PING (D2H AND H2D)
# ══════════════════════════════════════════════════════════════════════════════


def d2h_ping_plot(path="d2h_ping_iterations.csv", out="d2h_ping_timeseries.png"):
    p = _resolve_input_path(path)
    in_csv = p / "d2h_ping_iterations.csv" if p.is_dir() else p
    out_path = in_csv.parent / out
    try:
        df = pd.read_csv(in_csv)
    except FileNotFoundError:
        print(f"  Warning: {in_csv} not found")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.iteration, df.latency_us, "o-", ms=3, lw=1, alpha=0.7)
    ax.axhline(df.latency_us.median(), color="red", ls="--", lw=1.5, label=f"p50 = {df.latency_us.median():.2f} us")
    ax.axhline(df.latency_us.mean(), color="orange", ls=":", lw=1.5, label=f"avg = {df.latency_us.mean():.2f} us")
    ax.set(xlabel="Iteration", ylabel="Latency (us)", title="D2H Pure Ping: Per-Iteration Latency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, title="Series")
    _save_fig(fig, out_path)


def d2h_ping_export_csv(path="d2h_ping_iterations.csv", out="d2h_ping_summary.csv"):
    p = _resolve_input_path(path)
    in_csv = p / "d2h_ping_iterations.csv" if p.is_dir() else p
    out_path = in_csv.parent / out
    try:
        df = pd.read_csv(in_csv)
        df.to_csv(out_path, index=False, float_format="%.4f")
        print(f"  Saved {out_path} ({len(df)} iterations)")
    except FileNotFoundError:
        print(f"  Warning: {in_csv} not found, skipping CSV export")


def run_d2h_ping(path):
    d2h_ping_plot(path)
    d2h_ping_export_csv(path)


def h2d_ping_plot(csv_dir, out="h2d_ping_timeseries.png"):
    csv_dir = _resolve_input_path(csv_dir)
    modes = {"HOST_PUSH": None, "DEVICE_PULL": None}
    for mode in modes:
        p = csv_dir / f"h2d_ping_iterations_{mode}.csv"
        try:
            modes[mode] = pd.read_csv(p)
        except FileNotFoundError:
            print(f"  Warning: {p} not found")

    present = {k: v for k, v in modes.items() if v is not None}
    if not present:
        return

    out_path = csv_dir / out
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode, df in present.items():
        ax.plot(df.iteration, df.latency_us, "o-", ms=3, lw=1, alpha=0.7, label=mode)
        ax.axhline(
            df.latency_us.median(),
            ls="--",
            lw=1.5,
            alpha=0.6,
            label=f"{mode} p50 = {df.latency_us.median():.2f} us",
        )
    ax.set(xlabel="Iteration", ylabel="Latency (us)", title="H2D Pure Ping: Per-Iteration Latency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, title="Series")
    _save_fig(fig, out_path)


def h2d_ping_export_csv(csv_dir, out="h2d_ping_summary.csv"):
    csv_dir = _resolve_input_path(csv_dir)
    dfs = []
    for mode in ["HOST_PUSH", "DEVICE_PULL"]:
        p = csv_dir / f"h2d_ping_iterations_{mode}.csv"
        try:
            df = pd.read_csv(p)
            df["h2d_mode"] = mode
            dfs.append(df)
        except FileNotFoundError:
            print(f"  Warning: {p} not found, skipping")
    if not dfs:
        print("  Warning: no H2D ping iteration files found")
        return
    out_path = csv_dir / out
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Saved {out_path} ({len(combined)} rows)")


def run_h2d_ping(path):
    p = _resolve_input_path(path)
    csv_dir = p if p.is_dir() else p.parent
    h2d_ping_plot(csv_dir)
    h2d_ping_export_csv(csv_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-CHIP THROUGHPUT  (D2H or H2D)
# ══════════════════════════════════════════════════════════════════════════════


def mc_print_report(df, direction="D2H"):
    chips = df["chip_label"].unique()
    fifos = sorted(df["socket_fifo_size"].unique())

    print(f"\n{'='*70}")
    print(f"  {direction} Multi-Chip Max Throughput  ({len(df)} rows, {len(chips)} chip(s))")
    print(f"{'='*70}")
    print(f"  Page size  : {human_bytes(int(df.data_size.iloc[0]))} (fixed)")
    print(f"  Total data : {human_bytes(int(df.total_data.iloc[0]))}")
    print(f"  FIFO sizes : {[human_bytes(f) for f in fifos]}")
    print(f"  Chips      : {list(chips)}")

    print(f"\n  Peak throughput per chip:")
    print(f"  {'Chip':<36} {'FIFO':>8} {'GB/s':>8}")
    print(f"  {'-'*54}")
    for chip in chips:
        sub = df[df.chip_label == chip]
        best = sub.sort_values("throughput_gbps", ascending=False).iloc[0]
        print(f"  {chip:<36} {human_bytes(int(best.socket_fifo_size)):>8} " f"{best.throughput_gbps:>8.3f}")

    print(f"\n  Throughput (GB/s) by chip × FIFO size:")
    hdr = f"  {'Chip':<36}" + "".join(f"{human_bytes(f):>9}" for f in fifos)
    print(f"{hdr}\n  {'-'*(len(hdr)-2)}")
    for chip in chips:
        row = f"  {chip:<36}"
        sub = df[df.chip_label == chip]
        for fs in fifos:
            c = sub[sub.socket_fifo_size == fs]
            row += f"{c.throughput_gbps.values[0]:>9.3f}" if len(c) else f"{'--':>9}"
        print(row)

    fs_max = max(fifos)
    sub_max = df[df.socket_fifo_size == fs_max]
    print(f"\n  Per-page timing at FIFO={human_bytes(fs_max)}:")
    print(f"  {'Chip':<36} {'us/page':>10} {'cycles':>10} {'GB/s':>8}")
    print(f"  {'-'*66}")
    for chip in chips:
        r = sub_max[sub_max.chip_label == chip]
        if r.empty:
            continue
        r = r.iloc[0]
        print(f"  {chip:<36} {r.per_page_us:>10.3f} " f"{r.per_page_cycles:>10.0f} {r.throughput_gbps:>8.3f}")


def mc_plot_throughput_vs_fifo(df, out="mc_throughput_vs_fifo.png", direction="D2H"):
    fig, ax = plt.subplots(figsize=(11, 6))
    for chip in sorted(df.chip_label.unique()):
        g = df[df.chip_label == chip].sort_values("socket_fifo_size")
        ax.plot(g.socket_fifo_size, g.throughput_gbps, "o-", label=chip, ms=6, lw=2)
    fifos = sorted(df.socket_fifo_size.unique())
    page_size = human_bytes(int(df.data_size.iloc[0]))
    ax.set(
        xscale="log",
        xlabel="Socket FIFO Size",
        ylabel="GB/s",
        title=f"{direction} Max Throughput vs FIFO Size — All Chips (page={page_size}, total=1GB)",
    )
    _set_size_ticks(ax, fifos, rotation=30)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, title="Chip")
    _save_fig(fig, out)


def mc_plot_bar_comparison(df, out="mc_throughput_bar.png", direction="D2H"):
    chips = sorted(df.chip_label.unique())
    fifos = sorted(df.socket_fifo_size.unique())
    x = np.arange(len(fifos))
    width = 0.8 / max(len(chips), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, chip in enumerate(chips):
        sub = df[df.chip_label == chip].set_index("socket_fifo_size").reindex(fifos)
        ax.bar(x + i * width, sub.throughput_gbps, width, label=chip)

    page_size = human_bytes(int(df.data_size.iloc[0]))
    ax.set_xticks(x + width * (len(chips) - 1) / 2)
    ax.set_xticklabels([human_bytes(f) for f in fifos], rotation=30, fontsize=9)
    ax.set(
        xlabel="Socket FIFO Size",
        ylabel="GB/s",
        title=f"{direction} Max Throughput by Chip (page={page_size}, total=1GB)",
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, title="Chip")
    _save_fig(fig, out)


def mc_plot_heatmap(df, out="mc_throughput_heatmap.png", direction="D2H"):
    chips = sorted(df.chip_label.unique())
    fifos = sorted(df.socket_fifo_size.unique())

    data = np.full((len(chips), len(fifos)), np.nan)
    for i, chip in enumerate(chips):
        for j, fs in enumerate(fifos):
            c = df[(df.chip_label == chip) & (df.socket_fifo_size == fs)]
            if not c.empty:
                data[i, j] = c.throughput_gbps.values[0]

    fig, ax = plt.subplots(figsize=(max(8, len(fifos) * 1.4), max(4, len(chips) * 0.8 + 2)))
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(fifos)))
    ax.set_xticklabels([human_bytes(f) for f in fifos], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(chips)))
    ax.set_yticklabels(chips, fontsize=9)
    ax.set_xlabel("Socket FIFO Size", fontsize=10)
    ax.set_ylabel("Chip (Tray/ASIC)", fontsize=10)
    page_size = human_bytes(int(df.data_size.iloc[0]))
    ax.set_title(
        f"{direction} Throughput (GB/s) — All Chips × FIFO Size (page={page_size}, total=1GB)",
        fontsize=11,
    )
    _annotate_heatmap(ax, data, vmax, fontsize=9)

    fig.colorbar(im, ax=ax, label="GB/s", shrink=0.8)
    _save_fig(fig, out)


def mc_export_csv(df, out="mc_throughput_summary.csv"):
    pv = df.pivot_table(index="chip_label", columns="socket_fifo_size", values="throughput_gbps")
    pv.columns = [f"FIFO={human_bytes(c)} GB/s" for c in pv.columns]
    pv.index.name = "chip (tray/asic/coord)"
    us_pv = df.pivot_table(index="chip_label", columns="socket_fifo_size", values="per_page_us")
    us_pv.columns = [f"FIFO={human_bytes(c)} us/page" for c in us_pv.columns]
    us_pv.index.name = "chip (tray/asic/coord)"
    pv.join(us_pv).to_csv(out, float_format="%.6f")
    print(f"  Saved {out}")


def run_d2h_multichip(path, prefix="d2h_mc_"):
    df = load_gbench_d2h_multichip_csv(path)
    mc_print_report(df, direction="D2H")
    mc_plot_throughput_vs_fifo(df, out=f"{prefix}throughput_vs_fifo.png", direction="D2H")
    mc_plot_bar_comparison(df, out=f"{prefix}throughput_bar.png", direction="D2H")
    mc_plot_heatmap(df, out=f"{prefix}throughput_heatmap.png", direction="D2H")
    mc_export_csv(df, out=f"{prefix}throughput_summary.csv")


def run_h2d_multichip(path, prefix="h2d_mc_"):
    df = load_gbench_h2d_multichip_csv(path)
    mc_print_report(df, direction="H2D")
    mc_plot_throughput_vs_fifo(df, out=f"{prefix}throughput_vs_fifo.png", direction="H2D")
    mc_plot_bar_comparison(df, out=f"{prefix}throughput_bar.png", direction="H2D")
    mc_plot_heatmap(df, out=f"{prefix}throughput_heatmap.png", direction="H2D")
    mc_export_csv(df, out=f"{prefix}throughput_summary.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze H2D/D2H socket benchmark results")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--d2h-throughput", action="store_true", help="D2H throughput (gbench CSV)")
    mode.add_argument("--d2h-latency", action="store_true", help="D2H latency (gbench CSV)")
    mode.add_argument("--d2h-ping", action="store_true", help="D2H pure ping CSV / directory")
    mode.add_argument("--d2h-multichip", action="store_true", help="D2H multi-chip throughput (gbench CSV)")
    mode.add_argument("--h2d-throughput", action="store_true", help="H2D throughput (gbench CSV)")
    mode.add_argument("--h2d-latency", action="store_true", help="H2D latency (gbench CSV)")
    mode.add_argument("--h2d-ping", action="store_true", help="H2D pure ping CSV / directory")
    mode.add_argument("--h2d-multichip", action="store_true", help="H2D multi-chip throughput (gbench CSV)")
    mode.add_argument(
        "--gbench",
        action="store_true",
        help="Google Benchmark CSV — auto-routes to correct analysis by benchmark name prefix",
    )
    p.add_argument("csv", help="Path to benchmark CSV / log (or directory for ping modes)")
    p.add_argument(
        "--out-prefix",
        default="",
        metavar="PREFIX",
        help="Prefix prepended to every output file name (e.g. 'tray1_asic6_')",
    )
    args = p.parse_args()

    pfx = _auto_chip_prefix(args.csv, args.out_prefix)
    if args.gbench:
        run_gbench(args.csv, prefix=pfx)
    elif args.d2h_throughput:
        run_d2h_throughput(args.csv, prefix=pfx)
    elif args.d2h_latency:
        run_d2h_latency(args.csv, prefix=pfx)
    elif args.d2h_ping:
        run_d2h_ping(args.csv)
    elif args.d2h_multichip:
        run_d2h_multichip(args.csv, prefix=pfx or "d2h_mc_")
    elif args.h2d_throughput:
        run_h2d_throughput(args.csv, prefix=pfx)
    elif args.h2d_latency:
        run_h2d_latency(args.csv, prefix=pfx)
    elif args.h2d_ping:
        run_h2d_ping(args.csv)
    elif args.h2d_multichip:
        run_h2d_multichip(args.csv, prefix=pfx or "h2d_mc_")
