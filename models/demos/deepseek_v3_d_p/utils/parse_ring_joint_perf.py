# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Parse tracy profiler CSV output for RingJointSDPA perf tests.

Supports both single-host and multi-host runs. For multi-host, each rank
produces its own CSV with local device IDs 0-31; this script combines them
using (rank, device_id) as the unique device key.

Usage:
    # Single host (one CSV)
    python parse_ring_joint_perf.py <csv_path>

    # Multi-host (directory with rank0/, rank1/, ... subdirs containing CSVs)
    python parse_ring_joint_perf.py --report-dir <path_to_report_dir>

    # Multi-host (explicit CSV files, one per rank)
    python parse_ring_joint_perf.py --rank rank0.csv rank1.csv rank2.csv rank3.csv

Examples:
    # Single host
    python models/demos/deepseek_v3_d_p/utils/parse_ring_joint_perf.py generated/profiler/reports/2026_04_06_12_13_28/ops_perf_results_2026_04_06_12_13_28.csv

    # Multi-host (flat layout: rank*/ops_perf_results_*.csv)
    python models/demos/deepseek_v3_d_p/utils/parse_ring_joint_perf.py --report-dir models/demos/deepseek_v3_d_p/reports/ring_joint_perf

    # Multi-host (tt-run layout: rank*/reports/<timestamp>/ops_perf_results_*.csv)
    python models/demos/deepseek_v3_d_p/utils/parse_ring_joint_perf.py --report-dir /data/ipotkonjak/tt-metal/generated/profiler/ttrun
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

OP_NAME = "RingJointSDPADeviceOperation"
DURATION_COL = "DEVICE KERNEL DURATION [ns]"


def load_single_host(csv_paths):
    """Load one or more CSV files from a single host (device IDs are globally unique)."""
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    ops = combined[combined["OP CODE"] == OP_NAME].copy()
    ops["rank"] = "host"
    ops["device_label"] = "dev " + ops["DEVICE ID"].astype(int).astype(str)
    return ops


def load_multi_host_from_dir(report_dir):
    """
    Load CSVs from a report directory with rank*/ subdirs.

    Supports two layouts:
      1. rank*/ops_perf_results_*.csv  (flat, e.g. reports/ring_joint_perf/rank0/)
      2. rank*/reports/<timestamp_dir>/ops_perf_results_*.csv  (tt-run layout, e.g. generated/profiler/ttrun/rank0/reports/)

    In both cases, picks the latest CSV by filename timestamp.
    """
    report_path = Path(report_dir)
    rank_dirs = sorted(report_path.glob("rank*"))
    if not rank_dirs:
        print(f"ERROR: No rank*/ subdirectories found in {report_dir}", file=sys.stderr)
        sys.exit(1)

    csv_paths = []
    for rank_dir in rank_dirs:
        # Try flat layout first
        csvs = sorted(rank_dir.glob("ops_perf_results_*.csv"))
        if not csvs:
            # Try tt-run layout: rank*/reports/<timestamp>/ops_perf_results_*.csv
            csvs = sorted(rank_dir.glob("reports/*/ops_perf_results_*.csv"))
        if not csvs:
            print(f"WARNING: No CSV files found in {rank_dir}", file=sys.stderr)
            continue
        # Pick latest by filename (timestamp sorts lexicographically)
        csv_paths.append((rank_dir.name, csvs[-1]))

    if not csv_paths:
        print(f"ERROR: No CSV files found in any rank directory under {report_dir}", file=sys.stderr)
        sys.exit(1)

    return load_multi_host_ranked(csv_paths)


def load_multi_host_ranked(ranked_csv_paths):
    """Load CSVs with rank labels. ranked_csv_paths is list of (rank_label, csv_path)."""
    all_ops = []
    for rank_label, csv_path in ranked_csv_paths:
        df = pd.read_csv(csv_path)
        ops = df[df["OP CODE"] == OP_NAME].copy()
        ops["rank"] = rank_label
        ops["device_label"] = rank_label + "/dev " + ops["DEVICE ID"].astype(int).astype(str)
        all_ops.append(ops)

    if not all_ops:
        print("ERROR: No RingJointSDPA ops found in any CSV.", file=sys.stderr)
        sys.exit(1)

    return pd.concat(all_ops, ignore_index=True)


def parse_ring_joint_perf(ops):
    """
    Parse RingJointSDPA device kernel durations.

    Groups by device_label and uses cumcount to assign iteration indices:
      iteration 0 = compile run (or single run if only 1 iteration)
      iteration 1..N = perf runs

    Returns:
        ops: DataFrame with iteration column added
        num_iters: total number of iterations per device
    """
    if ops.empty:
        print(f"ERROR: No '{OP_NAME}' ops found in CSV.", file=sys.stderr)
        sys.exit(1)

    ops = ops.copy()
    ops["iteration"] = ops.groupby("device_label").cumcount()
    num_iters = ops["iteration"].max() + 1

    return ops, num_iters


def build_rank_iteration_tables(ops, num_iters):
    """
    Build tables of max and min device kernel duration per rank per iteration.

    Returns:
        max_table: DataFrame rows=iterations, columns=ranks, values=max duration
        max_devs: DataFrame same shape, values=device label of the worst device
        min_table: DataFrame rows=iterations, columns=ranks, values=min duration
        min_devs: DataFrame same shape, values=device label of the best device
    """
    ranks = sorted(ops["rank"].unique())

    max_table = {}
    max_devs = {}
    min_table = {}
    min_devs = {}
    for rank in ranks:
        rank_ops = ops[ops["rank"] == rank]
        max_table[rank] = rank_ops.groupby("iteration")[DURATION_COL].max()
        max_devs[rank] = rank_ops.loc[rank_ops.groupby("iteration")[DURATION_COL].idxmax()].set_index("iteration")[
            "device_label"
        ]
        min_table[rank] = rank_ops.groupby("iteration")[DURATION_COL].min()
        min_devs[rank] = rank_ops.loc[rank_ops.groupby("iteration")[DURATION_COL].idxmin()].set_index("iteration")[
            "device_label"
        ]

    return pd.DataFrame(max_table), pd.DataFrame(max_devs), pd.DataFrame(min_table), pd.DataFrame(min_devs)


def print_summary(ops, num_iters):
    """Print a human-readable summary of the perf results."""
    ranks = sorted(ops["rank"].unique())
    num_devices = ops["device_label"].nunique()
    is_multi_host = len(ranks) > 1

    max_table, max_devs, min_table, min_devs = build_rank_iteration_tables(ops, num_iters)

    if num_iters == 1:
        # Single iteration — just report the one run
        print(f"=== Single run across {num_devices} devices ({len(ranks)} rank(s)) ===")
        print()
        max_row = max_table.iloc[0]
        min_row = min_table.iloc[0]
        print(f"{'RANK':<10} {'MAX [ms]':>12} {'MIN [ms]':>12} {'SKEW [ms]':>12}  {'WORST DEVICE'}")
        print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12}  {'-'*20}")
        global_min = min_row.min()
        for rank in ranks:
            dev = max_devs.iloc[0][rank]
            dev_short = dev.split("/")[-1] if "/" in dev else dev
            skew = max_row[rank] - global_min
            print(
                f"{rank:<10} {max_row[rank] / 1e6:>12.3f} {min_row[rank] / 1e6:>12.3f} "
                f"{skew / 1e6:>12.3f}  {dev_short}"
            )
        print()
        global_max = max_row.max()
        global_max_rank = max_row.idxmax()
        print(f"  Global max:  {global_max / 1e6:.3f} ms ({max_devs.iloc[0][global_max_rank]})")
        print(f"  Global min:  {global_min / 1e6:.3f} ms")
        print(f"  Max skew:    {(global_max - global_min) / 1e6:.3f} ms")
        return

    num_perf_runs = num_iters - 1

    # --- Compile run ---
    compile_max_row = max_table.iloc[0]
    compile_max = compile_max_row.max()
    compile_max_rank = compile_max_row.idxmax()
    compile_worst_dev = max_devs.iloc[0][compile_max_rank]
    print(f"=== Compile run ===")
    print(f"  Max kernel duration: {compile_max / 1e6:.3f} ms ({compile_worst_dev})")
    print()

    # --- Per-iteration table: worst device kernel per rank ---
    perf_max_table = max_table.iloc[1:]
    perf_max_devs = max_devs.iloc[1:]

    header_rank = "".join(f"{r:>12}" for r in ranks)
    print(f"=== Worst device kernel duration per rank [ms] ===")
    print(f"{'':>12}{header_rank}")
    print(f"{'':>12}{'─' * (12 * len(ranks))}")

    for i, (idx, row) in enumerate(perf_max_table.iterrows()):
        run_num = i + 1
        row_str = "".join(f"{row[r] / 1e6:>12.3f}" for r in ranks)
        print(f"{'RUN ' + str(run_num):>12}{row_str}")

    print()

    # --- Worst device per rank per iteration (compact) ---
    if is_multi_host:
        print(f"=== Worst device per rank ===")
        header_rank_devs = "".join(f"{r:>16}" for r in ranks)
        print(f"{'':>12}{header_rank_devs}")
        print(f"{'':>12}{'─' * (16 * len(ranks))}")
        for i, (idx, row) in enumerate(perf_max_devs.iterrows()):
            run_num = i + 1
            row_str = ""
            for r in ranks:
                dev = row[r]
                dev_short = dev.split("/")[-1] if "/" in dev else dev
                row_str += f"{dev_short:>16}"
            print(f"{'RUN ' + str(run_num):>12}{row_str}")
        print()

    # --- Summary ---
    # For each iteration, take max and min of worst-device-per-rank across ranks
    # max = slowest rank (wall-clock wait), min = fastest rank (actual op time)
    iter_max = perf_max_table.max(axis=1)
    iter_min = perf_max_table.min(axis=1)
    iter_max.index = range(1, num_perf_runs + 1)
    iter_min.index = range(1, num_perf_runs + 1)
    iter_skew = iter_max - iter_min

    best_run = iter_max.idxmin()
    worst_run = iter_max.idxmax()

    # Find bottleneck devices
    worst_run_row = perf_max_table.iloc[worst_run - 1]
    worst_run_dev = perf_max_devs.iloc[worst_run - 1][worst_run_row.idxmax()]

    best_run_row = perf_max_table.iloc[best_run - 1]
    best_run_dev = perf_max_devs.iloc[best_run - 1][best_run_row.idxmax()]

    # Fastest rank = actual op time, slowest rank = wait time
    fastest_rank = perf_max_table.mean(axis=0).idxmin()
    slowest_rank = perf_max_table.mean(axis=0).idxmax()

    print(f"=== Summary ({num_perf_runs} perf runs, {num_devices} devices, {len(ranks)} rank(s)) ===")
    print(f"  Best run:  RUN {best_run}  {iter_max[best_run] / 1e6:.3f} ms  (bottleneck: {best_run_dev})")
    print(f"  Worst run: RUN {worst_run}  {iter_max[worst_run] / 1e6:.3f} ms  (bottleneck: {worst_run_dev})")
    print(f"  Avg max (wait time):   {iter_max.mean() / 1e6:.3f} ms  (slowest rank: {slowest_rank})")
    print(f"  Avg min (op time):     {iter_min.mean() / 1e6:.3f} ms  (fastest rank: {fastest_rank})")
    print(f"  Max skew:              {iter_skew.max() / 1e6:.3f} ms  (run {iter_skew.idxmax()})")
    print(f"  Spread:                {(iter_max.max() - iter_max.min()) / 1e3:.1f} us")


def main():
    parser = argparse.ArgumentParser(description="Parse RingJointSDPA perf test tracy CSV output")
    parser.add_argument("csv_paths", nargs="*", help="Path(s) to tracy ops_perf_results CSV file(s) (single host)")
    parser.add_argument(
        "--report-dir",
        help="Path to report directory with rank0/, rank1/, ... subdirs (multi-host)",
    )
    parser.add_argument(
        "--rank",
        nargs="+",
        help="Explicit per-rank CSV files in order: rank0.csv rank1.csv ... (multi-host)",
    )
    args = parser.parse_args()

    if args.report_dir:
        ops = load_multi_host_from_dir(args.report_dir)
    elif args.rank:
        ranked = [(f"rank{i}", path) for i, path in enumerate(args.rank)]
        ops = load_multi_host_ranked(ranked)
    elif args.csv_paths:
        ops = load_single_host(args.csv_paths)
    else:
        parser.print_help()
        sys.exit(1)

    ops, num_iters = parse_ring_joint_perf(ops)
    print_summary(ops, num_iters)


if __name__ == "__main__":
    main()
