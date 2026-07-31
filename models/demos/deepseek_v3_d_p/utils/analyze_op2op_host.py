#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Attribute device op-to-op latency to TTNN host dispatch stages.

This script consumes a Tracy profiler output directory containing:

* tracy_ops_data.csv
* tracy_ops_times.csv
* ops_perf_results_*.csv

It correlates the per-operation host zones in ``tracy_ops_times.csv`` with
device-0 rows in ``ops_perf_results_*.csv``. The first operation in the
selected iteration is excluded from op2op attribution because its
``OP TO OP LATENCY`` spans the gap between iterations.

Example:

    python models/demos/deepseek_v3_d_p/utils/analyze_op2op_host.py \
        kimi_l1_chunk0_16 --iteration 1
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HOST_STAGES = (
    "TTNN Op create output tensors",
    "TTNN Op compute mesh workload hash",
    "TTNN Op compute mesh canonical key",
    "TTNN Op program cache lookup",
    "TTNN Op validate cache hit",
    "TTNN Op update cached workload",
    "EnqueueProgram",
    # This aggregate contains the nested per-device TT_DNN_DEVICE_OP zones.
    "TTNN Op profile mesh workload",
)

DETAIL_ZONES = (
    # Nested inside "TTNN Op update cached workload"; report separately but do
    # not add to attributed totals, which would double-count their duration.
    "TTNN Op apply cached descriptor",
    "TTNN Op override runtime arguments",
    "TTNN Override apply per-program runtime args",
    "TTNN Descriptor collect tensor buffers",
    "TTNN Descriptor apply resolved bindings",
    "TTNN Descriptor apply dynamic runtime args",
    "TTNN Descriptor custom override runtime args",
    "TTNN Descriptor compute dynamic runtime args",
    "TTNN Descriptor rebuild runtime args",
    "TTNN RingJoint apply descriptor",
    "TTNN RingJoint apply scalar runtime args",
    "TTNN Rotary indexed apply descriptor",
    "TTNN Rotary indexed patch scalar runtime args",
    "TTNN RS copy reader runtime args",
    "TTNN RS copy writer runtime args",
)

SUPPORT_ZONES = (
    "CompileProgram",
    "FDMeshCommandQueue::finish",
    "FDMeshCommandQueue::finish_nolock",
)


@dataclass(frozen=True)
class Event:
    start_ns: int
    duration_ns: int


@dataclass(frozen=True)
class DeviceOp:
    start_ns: int
    name: str
    global_call_count: str
    op2op_ns: float
    kernel_ns: float


def find_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {filename} under {root}, found {len(matches)}: {matches}")
    return matches[0]


def find_ops_report(root: Path) -> Path:
    matches = sorted(root.rglob("ops_perf_results_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one ops_perf_results_*.csv under {root}, found {len(matches)}: {matches}")
    return matches[0]


def read_signposts(path: Path) -> dict[str, list[int]]:
    signposts: dict[str, list[int]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as file:
        next(file, None)
        for line in file:
            message, separator, timestamp = line.rstrip().rpartition(";")
            if not separator or "TT_SIGNPOST:" not in message:
                continue
            name = message.split("TT_SIGNPOST:", 1)[1].strip(' `"')
            signposts[name].append(int(timestamp))
    return signposts


def require_signpost(signposts: dict[str, list[int]], name: str) -> int:
    values = signposts.get(name, [])
    if len(values) != 1:
        raise RuntimeError(f"expected one {name!r} signpost, found {len(values)}")
    return values[0]


def read_host_events(path: Path, start_ns: int, end_ns: int) -> dict[str, list[Event]]:
    """Stream the multi-GB Tracy CSV, retaining only the small zone set we need."""
    wanted = set(HOST_STAGES) | set(DETAIL_ZONES) | set(SUPPORT_ZONES)
    events: dict[str, list[Event]] = defaultdict(list)

    with path.open(encoding="utf-8", errors="replace") as file:
        next(file, None)
        for line in file:
            # The retained zone names and source paths contain no commas before
            # ns_since_start, so a bounded split is substantially faster than
            # csv.DictReader on a multi-GB export.
            fields = line.rstrip().split(",", 8)
            if len(fields) < 7 or fields[0] not in wanted:
                continue
            try:
                timestamp = int(fields[5])
                duration = int(fields[6])
            except ValueError:
                continue
            if start_ns <= timestamp <= end_ns:
                events[fields[0]].append(Event(timestamp, duration))

    for zone_events in events.values():
        zone_events.sort(key=lambda event: event.start_ns)
    return events


def read_device_ops(path: Path, start_ns: int, end_ns: int, device_id: int) -> list[DeviceOp]:
    operations: list[DeviceOp] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as file:
        for row in csv.DictReader(file):
            try:
                timestamp = int(row["HOST START TS"])
                row_device_id = int(row["DEVICE ID"])
                op2op_ns = float(row["OP TO OP LATENCY [ns]"])
                kernel_ns = float(row["DEVICE KERNEL DURATION [ns]"])
            except (KeyError, TypeError, ValueError):
                continue
            if row_device_id == device_id and start_ns <= timestamp <= end_ns:
                operations.append(
                    DeviceOp(
                        start_ns=timestamp,
                        name=row["OP CODE"],
                        global_call_count=row["GLOBAL CALL COUNT"],
                        op2op_ns=op2op_ns,
                        kernel_ns=kernel_ns,
                    )
                )
    return sorted(operations, key=lambda operation: operation.start_ns)


def ns_to_ms(value: float) -> float:
    return value / 1e6


def print_stage_table(
    operations: list[DeviceOp],
    events: dict[str, list[Event]],
) -> tuple[float, float]:
    # Index 0 has an op2op value spanning the previous iteration, so both its
    # device gap and its host stages are excluded from attribution.
    attributed_operations = operations[1:]
    print("\nHost stage attribution (first operation excluded)")
    print(f"{'stage':42} {'count':>7} {'total ms':>11} {'mean us':>11} {'max ms':>10}")
    print("-" * 86)

    attributed_total = 0.0
    for stage in HOST_STAGES:
        stage_events = events.get(stage, [])
        if len(stage_events) != len(operations):
            raise RuntimeError(
                f"{stage!r}: expected {len(operations)} events to align with device ops, found {len(stage_events)}"
            )
        values = [event.duration_ns for event in stage_events[1:]]
        total = float(sum(values))
        attributed_total += total
        print(
            f"{stage:42} {len(values):7d} {ns_to_ms(total):11.3f} "
            f"{statistics.mean(values) / 1e3:11.3f} {ns_to_ms(max(values)):10.3f}"
        )

    device_op2op_total = sum(operation.op2op_ns for operation in attributed_operations)
    return attributed_total, device_op2op_total


def print_detail_zones(events: dict[str, list[Event]]) -> None:
    print("\nNested cached-workload detail (excluded from attribution total)")
    print(f"{'zone':42} {'count':>7} {'total ms':>11} {'mean us':>11} {'max ms':>10}")
    print("-" * 86)
    for zone in DETAIL_ZONES:
        values = [event.duration_ns for event in events.get(zone, [])]
        if not values:
            print(f"{zone:42} {0:7d} {0.0:11.3f} {0.0:11.3f} {0.0:10.3f}")
            continue
        print(
            f"{zone:42} {len(values):7d} {ns_to_ms(sum(values)):11.3f} "
            f"{statistics.mean(values) / 1e3:11.3f} {ns_to_ms(max(values)):10.3f}"
        )


def print_operation_table(operations: list[DeviceOp], events: dict[str, list[Event]], top: int) -> None:
    grouped: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    update_events = events["TTNN Op update cached workload"]

    for index, operation in enumerate(operations[1:], start=1):
        zoned_host_ns = sum(events[stage][index].duration_ns for stage in HOST_STAGES)
        values = grouped[operation.name]
        values[0] += 1
        values[1] += operation.op2op_ns
        values[2] += zoned_host_ns
        values[3] += update_events[index].duration_ns

    print(f"\nTop {top} operation types by device op2op")
    print(f"{'following operation':38} {'n':>3} {'op2op ms':>10} {'zoned ms':>10} {'resid ms':>10} {'update ms':>10}")
    print("-" * 88)
    ranked = sorted(grouped.items(), key=lambda item: item[1][1], reverse=True)
    for name, (count, op2op_ns, zoned_ns, update_ns) in ranked[:top]:
        print(
            f"{name:38} {int(count):3d} {ns_to_ms(op2op_ns):10.3f} "
            f"{ns_to_ms(zoned_ns):10.3f} {ns_to_ms(op2op_ns - zoned_ns):10.3f} "
            f"{ns_to_ms(update_ns):10.3f}"
        )


def print_largest_operation(operations: list[DeviceOp], events: dict[str, list[Event]]) -> None:
    # Exclude the first cross-iteration gap.
    index = max(range(1, len(operations)), key=lambda i: operations[i].op2op_ns)
    operation = operations[index]
    stage_total = sum(events[stage][index].duration_ns for stage in HOST_STAGES)

    print(f"\nLargest single gap: {operation.name} (global call {operation.global_call_count})")
    print(f"  device op2op: {ns_to_ms(operation.op2op_ns):.3f} ms")
    for stage in sorted(HOST_STAGES, key=lambda name: events[name][index].duration_ns, reverse=True):
        print(f"  {stage:40} {ns_to_ms(events[stage][index].duration_ns):8.3f} ms")
    print(f"  {'zoned host total':40} {ns_to_ms(stage_total):8.3f} ms")
    print(f"  {'unattributed residual':40} {ns_to_ms(operation.op2op_ns - stage_total):8.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiler_output", type=Path, help="Profiler output directory")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration signpost index to analyze (default: 1)")
    parser.add_argument("--device-id", type=int, default=0, help="Device row to use from ops report (default: 0)")
    parser.add_argument("--top", type=int, default=12, help="Number of operation types to print (default: 12)")
    args = parser.parse_args()

    root = args.profiler_output.resolve()
    ops_data_path = find_file(root, "tracy_ops_data.csv")
    ops_times_path = find_file(root, "tracy_ops_times.csv")
    ops_report_path = find_ops_report(root)

    signposts = read_signposts(ops_data_path)
    iteration_start = require_signpost(signposts, f"iter_{args.iteration}_start")
    iteration_end = require_signpost(signposts, f"iter_{args.iteration}_end")
    # Repeated layer signposts are expected when both iterations are present.
    layer_starts = signposts["forward_layer_0_start"]
    layer_ends = signposts["forward_layer_0_end"]
    if len(layer_starts) <= args.iteration or len(layer_ends) <= args.iteration:
        raise RuntimeError(f"missing layer signposts for iteration {args.iteration}")
    layer_start = layer_starts[args.iteration]
    layer_end = layer_ends[args.iteration]

    events = read_host_events(ops_times_path, iteration_start, iteration_end)
    operations = read_device_ops(ops_report_path, iteration_start, iteration_end, args.device_id)
    if len(operations) < 2:
        raise RuntimeError(f"need at least two device operations in iteration, found {len(operations)}")

    print(f"Profiler output: {root}")
    print(f"Iteration {args.iteration}: {ns_to_ms(iteration_end - iteration_start):.3f} ms")
    print(f"Layer 0: {ns_to_ms(layer_end - layer_start):.3f} ms")
    print(f"Device {args.device_id} operations: {len(operations)}")

    attributed_ns, device_op2op_ns = print_stage_table(operations, events)
    print_detail_zones(events)
    device_kernel_ns = sum(operation.kernel_ns for operation in operations)
    residual_ns = device_op2op_ns - attributed_ns

    print("\nEvidence summary")
    print(f"  device kernel time:             {ns_to_ms(device_kernel_ns):8.3f} ms")
    print(f"  device op2op (first excluded):  {ns_to_ms(device_op2op_ns):8.3f} ms")
    print(f"  zoned host stages:              {ns_to_ms(attributed_ns):8.3f} ms")
    print(f"  unattributed residual:          {ns_to_ms(residual_ns):8.3f} ms")
    print(f"  host-zone coverage of op2op:    {100.0 * attributed_ns / device_op2op_ns:8.1f} %")

    cache_hit_count = len(events.get("TTNN Op validate cache hit", []))
    compile_count = len(events.get("CompileProgram", []))
    in_layer_finishes = sum(
        layer_start <= event.start_ns <= layer_end
        for name in ("FDMeshCommandQueue::finish", "FDMeshCommandQueue::finish_nolock")
        for event in events.get(name, [])
    )
    print(f"  cache-hit-path operations:      {cache_hit_count:8d} / {len(operations)}")
    print(f"  CompileProgram zones:           {compile_count:8d}")
    print(f"  queue-finish zones in layer:    {in_layer_finishes:8d}")

    print_operation_table(operations, events, args.top)
    print_largest_operation(operations, events)


if __name__ == "__main__":
    main()
