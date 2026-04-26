#!/usr/bin/env python3
"""Analyze Tracy profiling output CSVs and print a summary report.

Usage:
    python scripts/analyze_profile.py
    python scripts/analyze_profile.py --logs-dir generated/profiler/.logs
    python scripts/analyze_profile.py --logs-dir generated/profiler/.logs --outlier-threshold-ms 2.0
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_LOGS_DIR = "generated/profiler/.logs"
DEFAULT_OUTLIER_THRESHOLD_MS = 1.0


def load_device_ops(device_csv: Path) -> list[dict]:
    ops = []
    with open(device_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ops.append(
                    {
                        "dur_ns": int(row.get("DEVICE KERNEL DURATION [ns]", "0") or "0"),
                        "cores": int(row.get("CORE COUNT", "0") or "0"),
                        "call_count": int(row.get("GLOBAL CALL COUNT", "0") or "0"),
                        "fw_dur_ns": int(row.get("DEVICE FW DURATION [ns]", "0") or "0"),
                        "op_name": row.get("OP NAME", ""),
                    }
                )
            except (ValueError, TypeError):
                continue
    return ops


def load_host_ops(host_csv: Path) -> dict[str, dict]:
    host_ops: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_ns": 0, "max_ns": 0})
    with open(host_csv) as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = row.get("MessageName", "")
            try:
                total_ns = int(row.get("total_ns", "0"))
            except (ValueError, TypeError):
                continue
            if "Profiler DRAM" in name or not name.strip() or name == "}`":
                continue
            if "TT_DNN_DEVICE_OP" in name:
                m = re.search(r'"(\w+)"', name)
                op_type = m.group(1) if m else name
            else:
                op_type = name.strip()
            host_ops[op_type]["count"] += 1
            host_ops[op_type]["total_ns"] += total_ns
            host_ops[op_type]["max_ns"] = max(host_ops[op_type]["max_ns"], total_ns)
    return dict(host_ops)


def print_device_summary(ops: list[dict], outlier_threshold_ns: int) -> None:
    outliers = [o for o in ops if o["dur_ns"] > outlier_threshold_ns]
    steady = [o for o in ops if o["dur_ns"] <= outlier_threshold_ns]

    if not steady:
        print("No steady-state device ops found.")
        return

    total_us = sum(o["dur_ns"] for o in steady) / 1000

    print(f"{'=' * 70}")
    print("DEVICE KERNEL PERF SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total device ops (steady-state): {len(steady)}")
    print(f"Total device kernel time: {total_us:.0f} us ({total_us / 1000:.2f} ms)")
    print(f"Avg per op: {total_us / len(steady):.1f} us")
    print(
        f"Outliers (> {outlier_threshold_ns / 1e6:.1f} ms): {len(outliers)} ops, "
        f"{sum(o['dur_ns'] for o in outliers) / 1000:.0f} us"
    )
    print()

    # By core count
    by_cores: dict[int, dict] = defaultdict(lambda: {"count": 0, "total_ns": 0, "max_ns": 0, "min_ns": float("inf")})
    for o in steady:
        c = o["cores"]
        by_cores[c]["count"] += 1
        by_cores[c]["total_ns"] += o["dur_ns"]
        by_cores[c]["max_ns"] = max(by_cores[c]["max_ns"], o["dur_ns"])
        by_cores[c]["min_ns"] = min(by_cores[c]["min_ns"], o["dur_ns"])

    total_ns_all = sum(o["dur_ns"] for o in steady)
    print(f"{'CORES':>5}  {'COUNT':>6}  {'TOTAL_us':>9}  {'AVG_us':>7}  {'MAX_us':>7}  {'%':>5}")
    print("-" * 55)
    for cores in sorted(by_cores.keys(), key=lambda c: by_cores[c]["total_ns"], reverse=True)[:20]:
        v = by_cores[cores]
        pct = v["total_ns"] / total_ns_all * 100
        print(
            f"{cores:5d}  {v['count']:6d}  {v['total_ns'] / 1000:9.0f}  "
            f"{v['total_ns'] / v['count'] / 1000:7.1f}  {v['max_ns'] / 1000:7.1f}  {pct:5.1f}"
        )

    # Top slowest
    print()
    print("TOP 15 SLOWEST STEADY-STATE KERNELS:")
    print(f"{'#':>3}  {'DUR_us':>8}  {'CORES':>5}  {'FW_us':>8}")
    print("-" * 35)
    for i, o in enumerate(sorted(steady, key=lambda x: x["dur_ns"], reverse=True)[:15]):
        print(f"{i + 1:3d}  {o['dur_ns'] / 1000:8.1f}  {o['cores']:5d}  {o['fw_dur_ns'] / 1000:8.1f}")

    # Histogram
    print()
    print("DURATION HISTOGRAM:")
    buckets = [(0, 1), (1, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500)]
    for lo, hi in buckets:
        count = sum(1 for o in steady if lo <= o["dur_ns"] / 1000 < hi)
        pct = count / len(steady) * 100
        bar = "#" * min(50, int(pct))
        print(f"  {lo:>4}-{hi:<4} us: {count:5d} ({pct:5.1f}%) {bar}")
    count_big = sum(1 for o in steady if o["dur_ns"] / 1000 >= 500)
    if count_big:
        print(f"   500+    us: {count_big:5d} ({count_big / len(steady) * 100:5.1f}%)")


def print_host_summary(host_ops: dict[str, dict]) -> None:
    print()
    print(f"{'=' * 70}")
    print("HOST OPS BY TYPE (op dispatch counts)")
    print(f"{'=' * 70}")
    print(f"Unique op types: {len(host_ops)}")
    print()
    print(f"{'COUNT':>6}  {'OP_TYPE'}")
    print("-" * 50)
    for name, v in sorted(host_ops.items(), key=lambda x: x[1]["count"], reverse=True)[:30]:
        print(f"{v['count']:6d}  {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Tracy profiling CSVs.")
    parser.add_argument(
        "--logs-dir",
        default=DEFAULT_LOGS_DIR,
        help=f"Path to profiler logs directory (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--outlier-threshold-ms",
        type=float,
        default=DEFAULT_OUTLIER_THRESHOLD_MS,
        help=f"Kernels above this duration are classified as outliers (default: {DEFAULT_OUTLIER_THRESHOLD_MS} ms)",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    device_csv = logs_dir / "cpp_device_perf_report.csv"
    host_csv = logs_dir / "tracy_ops_data.csv"

    if not device_csv.is_file():
        print(f"ERROR: Device perf CSV not found: {device_csv}", file=sys.stderr)
        print(f"Run a Tracy profile first, e.g.:", file=sys.stderr)
        print(f"  TT_METAL_DEVICE_PROFILER=1 python -m tracy -v -r -p -n <name> <script_or_-m_pytest>", file=sys.stderr)
        return 1

    outlier_threshold_ns = int(args.outlier_threshold_ms * 1_000_000)

    device_ops = load_device_ops(device_csv)
    print_device_summary(device_ops, outlier_threshold_ns)

    if host_csv.is_file():
        host_ops = load_host_ops(host_csv)
        print_host_summary(host_ops)
    else:
        print(f"\nWARNING: Host ops CSV not found: {host_csv}", file=sys.stderr)

    print()
    print(f"Raw data: {logs_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
