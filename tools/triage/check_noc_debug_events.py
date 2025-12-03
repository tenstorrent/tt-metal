#!/usr/bin/env python3

import json
import os
from pathlib import Path
from collections import defaultdict


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

    # Print out the locked ranges for each core
    for core, timeline in events.items():
        for event in timeline:
            if "currently_locked_ranges" not in event:
                continue
            print(f"{core} Locked ranges at {event['timestamp']}: {event['currently_locked_ranges']}")


if __name__ == "__main__":
    check_noc_debug_events()
