#!/usr/bin/env python3
"""Summarize opt-in all-gather device-cycle attribution records.

Run this after the safe pytest process exits, because worker profiler buffers
are materialized into profile_log_device.csv when the mesh device closes.
Intervals from different RISCs overlap and must not be added together.
"""

import argparse
import csv
import re
import statistics
from pathlib import Path


DEVICE_ID_NUM_BITS = 10
DEVICE_OP_ID_NUM_BITS = 31


def decode_base_runtime_id(encoded_runtime_id):
    return (encoded_runtime_id & ((1 << DEVICE_OP_ID_NUM_BITS) - 1)) >> DEVICE_ID_NUM_BITS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("generated/profiler/.logs/profile_log_device.csv"),
        help="device profiler CSV produced after the safe pytest process exits",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=7,
        help="use the last N all-gather runtime IDs (the perf test's timed samples)",
    )
    parser.add_argument(
        "--runtime-ids",
        help="comma-separated base runtime IDs; overrides --samples",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(f"device profiler log does not exist: {args.input}")

    with args.input.open(newline="") as csv_file:
        metadata = next(csv_file)
        frequency_match = re.search(r"CHIP_FREQ\[MHz\]:\s*([0-9.]+)", metadata)
        if not frequency_match:
            raise ValueError(f"missing chip frequency in profiler metadata: {metadata.strip()}")
        frequency_mhz = float(frequency_match.group(1))

        # Each interval is emitted once per worker/receiver RISC.  Device cores
        # execute concurrently, so retain values per core and find the critical
        # core separately for reader, writer, and receiver roles.
        per_core = {}
        for row in csv.DictReader(csv_file, skipinitialspace=True):
            zone = row["zone name"]
            if not zone.startswith("AG_") or row["type"] != "TS_DATA":
                continue
            runtime_id = decode_base_runtime_id(int(row["run host ID"]))
            role = (
                "reader"
                if zone.startswith(("AG_READER_", "AG_R_"))
                else "writer"
                if zone.startswith("AG_WRITER_")
                else "receiver"
            )
            core = (row["PCIe slot"], row["core_x"], row["core_y"], row["RISC processor type"])
            per_core.setdefault((runtime_id, role, core), {})[zone] = int(row["data"])

    available_runtime_ids = sorted({runtime_id for runtime_id, _, _ in per_core})
    if not available_runtime_ids:
        raise ValueError(f"no AG attribution records found in {args.input}")
    if args.runtime_ids:
        selected_runtime_ids = [int(value) for value in args.runtime_ids.split(",") if value]
    else:
        selected_runtime_ids = available_runtime_ids[-args.samples :]
    missing = sorted(set(selected_runtime_ids) - set(available_runtime_ids))
    if missing:
        raise ValueError(f"runtime IDs not present in attribution log: {missing}")

    print(
        f"AG_ATTRIBUTION_SUMMARY input={args.input} frequency_mhz={frequency_mhz:g} "
        f"runtime_ids={selected_runtime_ids}"
    )
    cycles_to_ms = 1.0 / frequency_mhz / 1e3
    for role in ("reader", "writer", "receiver"):
        role_runtime_ids = {runtime_id for runtime_id, candidate_role, _ in per_core if candidate_role == role}
        if not role_runtime_ids:
            continue
        critical_by_runtime = {}
        for runtime_id in selected_runtime_ids:
            candidates = [
                (
                    sum(value for zone, value in intervals.items() if zone.endswith("_CYCLES")),
                    sum(value for zone, value in intervals.items() if zone.endswith("_COUNT")),
                    core,
                    intervals,
                )
                for (candidate_runtime_id, candidate_role, core), intervals in per_core.items()
                if candidate_runtime_id == runtime_id and candidate_role == role
            ]
            if not candidates:
                raise ValueError(f"no {role} attribution records for runtime ID {runtime_id}")
            critical_by_runtime[runtime_id] = max(
                candidates,
                key=lambda candidate: (candidate[0] > 0, candidate[0] if candidate[0] > 0 else candidate[1]),
            )

        total_samples = [critical_by_runtime[runtime_id][0] for runtime_id in selected_runtime_ids]
        if any(total_samples):
            print(
                f"AG_ATTRIBUTION_ROLE role={role} critical_core_total_median="
                f"{statistics.median(total_samples) * cycles_to_ms:.6f}ms "
                f"critical_core_total_min={min(total_samples) * cycles_to_ms:.6f}ms "
                f"critical_core_total_max={max(total_samples) * cycles_to_ms:.6f}ms "
                f"samples_cycles={total_samples}"
            )
        else:
            count_scores = [critical_by_runtime[runtime_id][1] for runtime_id in selected_runtime_ids]
            print(
                f"AG_ADDRESS_ROLE role={role} critical_core_count_score_median="
                f"{statistics.median(count_scores):g} critical_core_count_score_min={min(count_scores)} "
                f"critical_core_count_score_max={max(count_scores)} samples={count_scores}"
            )
        zones = sorted({zone for _, _, _, intervals in critical_by_runtime.values() for zone in intervals})
        for zone in zones:
            samples = [critical_by_runtime[runtime_id][3].get(zone, 0) for runtime_id in selected_runtime_ids]
            if zone.endswith("_COUNT"):
                print(
                    f"AG_COMMAND_COUNT role={role} zone={zone} critical_role_core_median="
                    f"{statistics.median(samples):g} critical_role_core_min={min(samples)} "
                    f"critical_role_core_max={max(samples)} samples={samples}"
                )
            else:
                print(
                    f"AG_ATTRIBUTION role={role} zone={zone} critical_role_core_median="
                    f"{statistics.median(samples) * cycles_to_ms:.6f}ms "
                    f"critical_role_core_min={min(samples) * cycles_to_ms:.6f}ms "
                    f"critical_role_core_max={max(samples) * cycles_to_ms:.6f}ms samples_cycles={samples}"
                )


if __name__ == "__main__":
    main()
