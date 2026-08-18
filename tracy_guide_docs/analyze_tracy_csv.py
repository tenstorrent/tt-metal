#!/usr/bin/env python3
"""Summarize a tt-metal ops_perf_results CSV without external dependencies."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def milliseconds(nanoseconds: float) -> str:
    return f"{nanoseconds / 1e6:.3f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--session",
        default="auto",
        help="Metal trace replay session to analyze; default: highest numeric session",
    )
    parser.add_argument("--top", type=int, default=15, help="Number of operation groups to print")
    args = parser.parse_args()

    with args.csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    print(f"Source: {args.csv_path}")
    print(f"Rows: {len(rows):,}")
    print(f"Operation types: {dict(Counter(row['OP TYPE'] for row in rows))}")

    device_rows = [row for row in rows if row["DEVICE ID"]]
    devices = sorted({row["DEVICE ID"] for row in device_rows}, key=int)
    print(f"Devices: {devices}")

    signposts = [(index, row) for index, row in enumerate(rows) if row["OP TYPE"] == "signpost"]
    starts = [(index, row) for index, row in signposts if row["OP CODE"] == "start"]
    stops = [(index, row) for index, row in signposts if row["OP CODE"] == "stop"]
    if starts and stops:
        start_index, start_row = starts[-1]
        stop_index, stop_row = next(
            (item for item in stops if item[0] > start_index),
            stops[-1],
        )
        start_ns = number(start_row["HOST START TS"])
        stop_ns = number(stop_row["HOST START TS"])
        print(f"Final signposted window: {milliseconds(stop_ns - start_ns)}")
        print(f"CSV rows between signposts: {stop_index - start_index - 1:,}")

    replay_counts = Counter(
        row["METAL TRACE REPLAY SESSION ID"] for row in device_rows if row["METAL TRACE REPLAY SESSION ID"]
    )
    print(f"Trace replay sessions: {dict(sorted(replay_counts.items()))}")

    session = args.session
    if session == "auto" and replay_counts:
        session = max(replay_counts, key=lambda value: int(value))

    if session != "auto" and replay_counts:
        measured = [row for row in device_rows if row["METAL TRACE REPLAY SESSION ID"] == session]
        print(f"Analyzing trace replay session: {session}")
    elif starts and stops:
        measured = [row for row in rows[start_index + 1 : stop_index] if row["OP TYPE"] == "tt_dnn_device"]
        print("Analyzing device operations between the final signposts")
    else:
        measured = device_rows
        print("No replay session or signposted window found; analyzing the full CSV")

    measured_devices = sorted({row["DEVICE ID"] for row in measured}, key=int)
    per_device_total: dict[str, float] = defaultdict(float)
    per_device_op: dict[tuple[str, str], float] = defaultdict(float)
    op_counts = Counter()

    for row in measured:
        duration = number(row["DEVICE KERNEL DURATION [ns]"])
        if math.isnan(duration):
            continue
        device = row["DEVICE ID"]
        op_code = row["OP CODE"]
        per_device_total[device] += duration
        per_device_op[(device, op_code)] += duration
        op_counts[op_code] += 1

    print(f"Measured rows: {len(measured):,}")
    print("\nKernel-duration sum per device:")
    for device in measured_devices:
        print(f"  device {device:>2}: {milliseconds(per_device_total[device])}")

    critical_device_total = max(per_device_total.values(), default=0.0)
    grouped_operations = []
    for op_code in op_counts:
        device_durations = [per_device_op[(device, op_code)] for device in measured_devices]
        critical_duration = max(device_durations, default=0.0)
        percentage = 100 * critical_duration / critical_device_total if critical_device_total else 0
        grouped_operations.append((critical_duration, percentage, op_code, op_counts[op_code]))

    print("\nOperation contributions (maximum per-device sum):")
    for duration, percentage, op_code, count in sorted(grouped_operations, reverse=True)[: args.top]:
        print(f"  {op_code:42} {milliseconds(duration):>12}  {percentage:5.1f}%  rows={count:,}")

    top_individual = sorted(
        measured,
        key=lambda row: number(row["DEVICE KERNEL DURATION [ns]"]),
        reverse=True,
    )[: args.top]
    print("\nLongest individual device operations:")
    for row in top_individual:
        duration = number(row["DEVICE KERNEL DURATION [ns]"])
        print(
            f"  device {row['DEVICE ID']:>2}  {row['OP CODE']:38} "
            f"{milliseconds(duration):>12}  call={row['GLOBAL CALL COUNT']}"
        )


if __name__ == "__main__":
    main()
