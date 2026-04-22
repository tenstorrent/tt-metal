#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Analyze DeviceZoneScopedN profiling data for NeighborPadAsync kernels.

Usage:
    python analyze_neighbor_pad_zones.py <profiler_dir> [--op-id <id>] [--all] [--per-core]

Where <profiler_dir> contains:
    profile_log_device.csv      — per-core device zone timings
    ops_perf_results_*.csv      — per-op host/device timing summary

The script identifies NeighborPadAsync op invocations, extracts NP-* zone
durations from the device log, and prints a breakdown by zone.

Zones instrumented:
  local_copy_reader   : NP-LC-READ  — DRAM read of local interior sticks
  local_copy_writer   : NP-LC-ZERO  — T-front zero-fill
                        NP-LC-COPY  — CB→DRAM copy of local interior data
                        NP-LC-SIG   — Phase 2 semaphore signal
  minimal_default_reader: NP-MDR-LOOP — outer_dim loop (read own/edge data)
                          NP-MDR-WAIT — wait for incoming H halo from fabric
  minimal_default_writer: NP-MDW-BARR — startup barrier (Phase 2 sync)
                          NP-MDW-LOOP — main outer_dim loop (own pad + fabric write)
                          NP-MDW-RECV — write incoming halo from L1→DRAM
                          NP-MDW-SIG  — Phase 2 signal
  phase2_w_reader     : NP-P2-BARR  — wait for Phase 1 barrier
                        NP-P2-LOOP  — read boundary sticks from DRAM
                        NP-P2-WAIT  — wait for incoming W neighbor semaphore
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import numpy as np


ZONE_PREFIX = "NP-"
NP_ZONES = [
    "NP-LC-READ",
    "NP-LC-ZERO",
    "NP-LC-COPY",
    "NP-LC-SIG",
    "NP-MDR-LOOP",
    "NP-MDR-WAIT",
    "NP-MDW-BARR",
    "NP-MDW-LOOP",
    "NP-MDW-RECV",
    "NP-MDW-SIG",
    "NP-P2-BARR",
    "NP-P2-LOOP",
    "NP-P2-WAIT",
]

ZONE_KERNEL_MAP = {
    "NP-LC-READ":  "local_copy_reader",
    "NP-LC-ZERO":  "local_copy_writer",
    "NP-LC-COPY":  "local_copy_writer",
    "NP-LC-SIG":   "local_copy_writer",
    "NP-MDR-LOOP": "minimal_default_reader",
    "NP-MDR-WAIT": "minimal_default_reader",
    "NP-MDW-BARR": "minimal_default_writer",
    "NP-MDW-LOOP": "minimal_default_writer",
    "NP-MDW-RECV": "minimal_default_writer",
    "NP-MDW-SIG":  "minimal_default_writer",
    "NP-P2-BARR":  "phase2_w_reader",
    "NP-P2-LOOP":  "phase2_w_reader",
    "NP-P2-WAIT":  "phase2_w_reader",
}


def load_chip_freq_mhz(device_log_path: str) -> float:
    with open(device_log_path) as f:
        header = f.readline()
    # Header: "ARCH: blackhole, CHIP_FREQ[MHz]: 1350, ..."
    for part in header.split(","):
        part = part.strip()
        if part.startswith("CHIP_FREQ"):
            return float(part.split(":")[1].strip())
    return 1000.0  # fallback


def load_device_log(device_log_path: str) -> pd.DataFrame:
    df = pd.read_csv(device_log_path, skiprows=1, header=0)
    df.columns = [c.strip() for c in df.columns]
    # Rename for convenience
    df = df.rename(columns={
        "PCIe slot": "chip",
        "core_x": "core_x",
        "core_y": "core_y",
        "RISC processor type": "risc",
        "time[cycles since reset]": "cycles",
        "run host ID": "run_host_id",
        "zone name": "zone_name",
        "type": "zone_type",
    })
    return df


def load_ops_perf(ops_perf_path: str) -> pd.DataFrame:
    df = pd.read_csv(ops_perf_path, header=0)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_ops_perf_csv(profiler_dir: str) -> str:
    pattern = str(Path(profiler_dir) / "ops_perf_results_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No ops_perf_results_*.csv found in {profiler_dir}")
    return matches[0]


def compute_zone_durations(df: pd.DataFrame, run_host_ids: list[int], freq_mhz: float) -> pd.DataFrame:
    """
    For each (run_host_id, chip, core_x, core_y, risc, zone_name), pair
    ZONE_START and ZONE_END rows and compute duration in ns.
    Returns a DataFrame with one row per zone invocation.
    """
    mask = (
        df["run_host_id"].isin(run_host_ids) &
        df["zone_name"].str.startswith(ZONE_PREFIX, na=False)
    )
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    group_cols = ["run_host_id", "chip", "core_x", "core_y", "risc", "zone_name"]
    starts = sub[sub["zone_type"] == "ZONE_START"].set_index(group_cols)["cycles"]
    ends   = sub[sub["zone_type"] == "ZONE_END"].set_index(group_cols)["cycles"]

    # Handle multiple invocations of same zone on same core (shouldn't happen for our outer zones,
    # but handle it by taking first start and last end per group)
    start_df = (
        sub[sub["zone_type"] == "ZONE_START"]
        .groupby(group_cols)["cycles"].min()
        .rename("start_cycles")
    )
    end_df = (
        sub[sub["zone_type"] == "ZONE_END"]
        .groupby(group_cols)["cycles"].max()
        .rename("end_cycles")
    )

    joined = start_df.to_frame().join(end_df, how="inner")
    joined["duration_cycles"] = joined["end_cycles"] - joined["start_cycles"]
    joined["duration_ns"] = joined["duration_cycles"] / freq_mhz * 1000.0
    joined = joined.reset_index()
    return joined


def print_zone_summary(durations: pd.DataFrame, op_fw_duration_ns: float | None = None):
    print("\n" + "=" * 78)
    print(f"{'Zone':<18} {'Kernel':<28} {'N cores':>7}  {'Mean ns':>10}  {'Max ns':>10}  {'P50 ns':>10}")
    print("-" * 78)

    # Aggregate across all op invocations and all cores per zone
    agg = (
        durations.groupby("zone_name")["duration_ns"]
        .agg(n_cores="count", mean="mean", max="max", p50=lambda x: x.quantile(0.5))
        .reset_index()
    )

    # Sort by max duration descending
    agg = agg.sort_values("max", ascending=False)

    for _, row in agg.iterrows():
        zone = row["zone_name"]
        kernel = ZONE_KERNEL_MAP.get(zone, "?")
        print(
            f"{zone:<18} {kernel:<28} {int(row['n_cores']):>7}  "
            f"{row['mean']:>10,.0f}  {row['max']:>10,.0f}  {row['p50']:>10,.0f}"
        )

    print("=" * 78)

    # Dominant zone analysis
    max_by_zone = agg.set_index("zone_name")["max"]
    if op_fw_duration_ns:
        print(f"\nOp FW duration (wall): {op_fw_duration_ns:,.0f} ns")
    top_zone = max_by_zone.idxmax()
    print(f"Slowest zone (max across cores): {top_zone}  ({max_by_zone[top_zone]:,.0f} ns)")

    # Group by kernel
    print("\nPer-kernel max (slowest core in each kernel):")
    durations["kernel"] = durations["zone_name"].map(ZONE_KERNEL_MAP)
    kernel_max = durations.groupby("kernel")["duration_ns"].max().sort_values(ascending=False)
    for kernel, ns in kernel_max.items():
        print(f"  {kernel:<30} {ns:>12,.0f} ns")


def print_per_core_breakdown(durations: pd.DataFrame, top_n: int = 10):
    print(f"\nPer-core breakdown (top {top_n} slowest cores per zone):")
    for zone in sorted(durations["zone_name"].unique()):
        sub = durations[durations["zone_name"] == zone].nlargest(top_n, "duration_ns")
        if sub.empty:
            continue
        print(f"\n  {zone}:")
        print(f"    {'chip':>5} {'core_x':>7} {'core_y':>7} {'risc':<10} {'ns':>12}")
        for _, r in sub.iterrows():
            print(f"    {int(r['chip']):>5} {int(r['core_x']):>7} {int(r['core_y']):>7} {r['risc']:<10} {r['duration_ns']:>12,.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiler_dir", help="Path to profiler report directory")
    parser.add_argument("--op-id", type=int, default=None,
                        help="Specific GLOBAL CALL COUNT (run_host_id) to analyze. "
                             "Default: use the slowest NeighborPadAsync entry.")
    parser.add_argument("--all", action="store_true",
                        help="Analyze ALL NeighborPadAsync invocations (default: only the slowest)")
    parser.add_argument("--per-core", action="store_true",
                        help="Print per-core breakdown for each zone")
    args = parser.parse_args()

    profiler_dir = args.profiler_dir
    device_log_path = str(Path(profiler_dir) / "profile_log_device.csv")
    ops_perf_path = find_ops_perf_csv(profiler_dir)

    print(f"Loading device log: {device_log_path}")
    freq_mhz = load_chip_freq_mhz(device_log_path)
    print(f"  Chip frequency: {freq_mhz} MHz")

    device_log = load_device_log(device_log_path)
    print(f"  Rows: {len(device_log):,}")

    print(f"Loading ops perf: {ops_perf_path}")
    ops_df = load_ops_perf(ops_perf_path)

    np_ops = ops_df[ops_df["OP CODE"].str.contains("NeighborPad", na=False)].copy()
    if np_ops.empty:
        print("ERROR: No NeighborPadAsync ops found in ops_perf CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(np_ops)} NeighborPadAsync rows across {np_ops['GLOBAL CALL COUNT'].nunique()} unique invocations")

    # Determine which op id(s) to analyze
    if args.op_id is not None:
        run_ids = [args.op_id]
        op_fw_ns = None
        matching = np_ops[np_ops["GLOBAL CALL COUNT"] == args.op_id]
        if not matching.empty:
            op_fw_ns = float(matching["DEVICE FW DURATION [ns]"].max())
        print(f"\nAnalyzing specified op id: {args.op_id}  (FW: {op_fw_ns:,.0f} ns)" if op_fw_ns else f"\nAnalyzing op id: {args.op_id}")
    elif args.all:
        run_ids = np_ops["GLOBAL CALL COUNT"].unique().tolist()
        op_fw_ns = None
        print(f"\nAnalyzing ALL {len(run_ids)} NeighborPadAsync invocations")
    else:
        # Pick the single slowest invocation (max DEVICE FW DURATION across all devices for each GLOBAL CALL COUNT)
        worst_row = np_ops.loc[np_ops["DEVICE FW DURATION [ns]"].idxmax()]
        worst_id = int(worst_row["GLOBAL CALL COUNT"])
        op_fw_ns = float(worst_row["DEVICE FW DURATION [ns]"])
        run_ids = [worst_id]
        print(f"\nSlowest NeighborPadAsync: GLOBAL CALL COUNT={worst_id}, FW={op_fw_ns:,.0f} ns")
        # Print shape info
        shape_cols = [c for c in np_ops.columns if "INPUT_0" in c or "OUTPUT_0" in c]
        shape_info = worst_row[shape_cols].dropna()
        if not shape_info.empty:
            print("  Input/output shape attributes:")
            for col, val in shape_info.items():
                print(f"    {col}: {val}")

    # Extract zone durations
    print(f"\nExtracting NP-* zones for run_host_id(s): {run_ids[:5]}{'...' if len(run_ids) > 5 else ''}")
    durations = compute_zone_durations(device_log, run_ids, freq_mhz)

    if durations.empty:
        print("No NP-* zone data found. Did you rebuild with TT_METAL_DEVICE_PROFILER=1 "
              "after adding DeviceZoneScopedN to the kernels?")
        sys.exit(1)

    print(f"  Found {len(durations):,} zone duration records across "
          f"{durations['zone_name'].nunique()} zone names, "
          f"{durations[['chip','core_x','core_y','risc']].drop_duplicates().shape[0]} cores")

    print_zone_summary(durations, op_fw_duration_ns=op_fw_ns)

    if args.per_core:
        print_per_core_breakdown(durations)


if __name__ == "__main__":
    main()
