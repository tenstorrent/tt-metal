#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from collections import defaultdict


# File path derived from profiler


def get_profiler_artifacts_dir() -> Path:
    """Get the profiler artifacts directory, mirroring C++ get_profiler_artifacts_dir()."""
    if profiler_dir := os.environ.get("TT_METAL_PROFILER_DIR"):
        return Path(profiler_dir)
    prefix = Path(os.environ.get("TT_METAL_HOME", ""))
    return prefix / "generated" / "profiler"


def get_profiler_logs_dir() -> Path:
    """Get the profiler logs directory, mirroring C++ get_profiler_logs_dir()."""
    return get_profiler_artifacts_dir() / ".logs"


def get_noc_events_report_path() -> Path:
    """
    Get the NOC events report path, mirroring the C++ implementation in profiler.cpp.

    Resolution order:
    1. TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH environment variable
    2. Fallback to profiler logs directory ($TT_METAL_HOME/generated/profiler/.logs/)
    """
    if noc_events_path := os.environ.get("TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH"):
        return Path(noc_events_path)
    return get_profiler_logs_dir()


def get_noc_event_reports_for_device(device_id: int) -> list[Path]:
    """
    Get all NOC event report files for a given device ID.

    Report file formats (from profiler.cpp):
    - noc_trace_dev{device_id}_{op_name}_ID{runtime_id}.json (with op_name)
    - noc_trace_dev{device_id}_ID{runtime_id}.json (without op_name)
    """
    noc_events_dir = get_noc_events_report_path()
    if not noc_events_dir.exists():
        return []

    # Match pattern: noc_trace_dev{device_id}_*.json
    pattern = f"noc_trace_dev{device_id}_*.json"
    return sorted(noc_events_dir.glob(pattern))


def load_noc_event_reports_for_device(device_id: int) -> list[dict]:
    """Load and parse all NOC event report JSON files for a given device ID."""
    reports = []
    for report_path in get_noc_event_reports_for_device(device_id):
        with open(report_path, "r") as f:
            report_data = json.load(f)
            reports.append(report_data)
    return reports


# class NocDebugTimeline:


def check_noc_debug_events(device_id: int = 0):
    noc_events_dir = get_noc_events_report_path()
    print(f"NOC events report path: {noc_events_dir}")

    report_files = get_noc_event_reports_for_device(device_id)
    print(f"Found {len(report_files)} report(s) for device {device_id}:")
    for report_file in report_files:
        print(f"  - {report_file.name}")

    reports = load_noc_event_reports_for_device(device_id)

    # Create event timeline for each core
    # key: (sx, sy)
    # value: events
    #
    # each event dict contains the locked address ranges at that time
    events: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for report in reports:
        for event in report:
            core = (event["sx"], event["sy"])
            events[core].append(event)

    # Iterate over event timeline
    # At each event, save a state of locked memory ranges for each core
    for core, timeline in events.items():
        currently_locked_ranges: list[tuple[int, int]] = []
        for event in timeline:
            if "type" in event and event["type"] == "MEM_LOCK":
                start = event["locked_address_base"]
                end = start + event["bytes_locked"]
                # Add new range and merge with any overlapping ranges
                new_ranges = []
                for r in currently_locked_ranges:
                    if end < r[0] or start > r[1]:
                        new_ranges.append(r)
                    else:
                        start = min(r[0], start)
                        end = max(r[1], end)
                new_ranges.append((start, end))
                currently_locked_ranges = new_ranges
            elif "type" in event and event["type"] == "MEM_UNLOCK":
                # Remove the region, potentially splitting existing ranges
                unlock_start = event["locked_address_base"]
                unlock_end = unlock_start + event["bytes_locked"]
                new_ranges = []
                for r in currently_locked_ranges:
                    if unlock_end <= r[0] or unlock_start >= r[1]:
                        new_ranges.append(r)
                    else:
                        if r[0] < unlock_start:
                            new_ranges.append((r[0], unlock_start))
                        # Keep portion after unlock region (if non-empty)
                        if r[1] > unlock_end:
                            new_ranges.append((unlock_end, r[1]))
                currently_locked_ranges = new_ranges
            event["currently_locked_ranges"] = currently_locked_ranges

    # for core, timeline in events.items():
    #     for event in timeline:
    #         if "currently_locked_ranges" not in event:
    #             continue
    #         print(f"{core} Locked ranges at {event['timestamp']}: {event['currently_locked_ranges']}")

    # Check if writes from one core occur during another core's locked period
    check_writes_during_lock(events, write_core=(2, 1), lock_core=(1, 1))


def get_locked_ranges_at_timestamp(
    events: dict[tuple[int, int], list[dict]], core: tuple[int, int], timestamp: int
) -> list[tuple[int, int]]:
    if core not in events:
        return []

    # Find the event with the closest timestamp <= the given timestamp
    locked_ranges = []
    for event in events[core]:
        event_ts = event.get("timestamp", 0)
        if event_ts > timestamp:
            break
        if "currently_locked_ranges" in event:
            locked_ranges = event["currently_locked_ranges"]

    return locked_ranges


def is_address_in_locked_range(addr: int, locked_ranges: list[tuple[int, int]]) -> tuple[bool, tuple[int, int] | None]:
    for locked_range in locked_ranges:
        if locked_range[0] <= addr < locked_range[1]:
            return True, locked_range
    return False, None


def check_writes_during_lock(
    events: dict[tuple[int, int], list[dict]], write_core: tuple[int, int], lock_core: tuple[int, int]
):
    if lock_core not in events:
        print(f"Lock core {lock_core} not found in events")
        return

    if write_core not in events:
        print(f"Write core {write_core} not found in events")
        return

    print(f"Checking writes from {write_core} to {lock_core} for locked address violations...")

    violations_found = 0
    writes_to_lock_core = 0

    for event in events[write_core]:
        if "type" in event and event["type"] == "WRITE_":
            timestamp = event["timestamp"]
            dst_addr = event.get("dst_addr")
            dst_core = (event.get("dx"), event.get("dy"))

            # Only check writes destined for the lock_core
            if dst_core != lock_core:
                continue

            writes_to_lock_core += 1

            # Get the locked ranges on lock_core at this timestamp
            locked_ranges = get_locked_ranges_at_timestamp(events, lock_core, timestamp)

            if not locked_ranges:
                continue

            # Check if the destination address falls within a locked range
            is_locked, matching_range = is_address_in_locked_range(dst_addr, locked_ranges)

            if is_locked:
                violations_found += 1
                print(
                    f"  VIOLATION: Write from {write_core} at timestamp {timestamp} "
                    f"to address {dst_addr} (0x{dst_addr:x}) on {lock_core} "
                    f"hits locked range {matching_range} (0x{matching_range[0]:x}-0x{matching_range[1]:x})"
                )

    print(f"Total writes from {write_core} to {lock_core}: {writes_to_lock_core}")
    if violations_found == 0:
        print(f"No locked address violations found")
    else:
        print(f"Found {violations_found} write(s) to locked addresses!")


if __name__ == "__main__":
    check_noc_debug_events()
