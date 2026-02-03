#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_callstacks [--all-cores] [--per-core]

Options:
    --all-cores        Show all cores including ones with Go Message = DONE. By default, DONE cores are filtered out.
    --per-core         Show one row per core (no aggregation). By default, rows are aggregated by canonical
                       callstack and Dev:Locs are range-compressed.

Description:
    Dumps callstacks for all devices in the system and for every supported risc processor.

    By default:
    - Filters out cores with DONE status (use --all-cores to include them)
    - Aggregates rows by canonical callstack (use --per-core for detailed per-core view)
    - Shows compressed Dev:Loc ranges (e.g., "0-31:1-0..2")

    Use --per-core to get the original one-row-per-core table with full details.
    Use -v/-vv to show more columns (affects per-core mode only).

    Color output is automatically enabled when stdout is a TTY (terminal) and can be overridden
    with TT_TRIAGE_COLOR environment variable (0=disable, 1=enable).

Owner:
    tt-vjovanovic
"""

from dataclasses import dataclass
from collections import defaultdict

from triage import ScriptConfig, log_check_risc, run_script, triage_field, collection_serializer
from callstack_provider import (
    run as get_callstack_provider,
    CallstackProvider,
    CallstacksData,
    KernelCallstackWithMessage,
    format_callstack_with_message,
)
from run_checks import run as get_run_checks
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.umd_device import TimeoutDeviceRegisterError
from ttexalens.hardware.risc_debug import CallstackEntry

script_config = ScriptConfig(
    depends=["run_checks", "callstack_provider"],
)


@dataclass
class AggregatedCallstackRow:
    """Aggregated view of callstacks grouped by canonical callstack."""

    # Kernel name (should be unique per canonical callstack)
    kernel_name: str | None = triage_field("Kernel Name")

    # Full formatted callstack (frames and/or error text)
    callstack: str = triage_field("Callstack")

    # Number of cores that share this canonical callstack
    num_callstacks: int = triage_field("#Callstacks")

    # Distinct RISC names (e.g. "brisc,trisc0,trisc1")
    risc_names: str = triage_field("RiscV")

    # v3-compressed Dev:x-y list, rendered as comma-separated
    devices_locs: list[str] = triage_field("Dev:Locs", collection_serializer(", "))


def _canonical_frame_key(frame: CallstackEntry) -> tuple[str, str, int]:
    """Extract canonical key from a callstack frame (function, file, line)."""
    fn = frame.function_name or ""
    file = frame.file or ""
    line = frame.line if frame.line is not None else -1
    return (fn, file, line)


def _canonical_callstack_key(kcwm: KernelCallstackWithMessage) -> tuple:
    """
    Generate a canonical key for grouping callstacks.

    - If there are frames: tuple of (fn,file,line) for each frame.
    - If no frames but message is set: ("__ERROR__", message).
    - If no frames and no message: ("__EMPTY__",).
    """
    if kcwm.callstack:
        return tuple(_canonical_frame_key(f) for f in kcwm.callstack)
    if kcwm.message:
        return ("__ERROR__", kcwm.message)
    return ("__EMPTY__",)


def _compress_devices_locs_v3(dev_loc_keys: list[tuple[int, int, int]]) -> list[str]:
    """
    Compress device:location pairs into range notation.

    Args:
        dev_loc_keys: list of (Dev, A, B) where A,B are noc0 coords.

    Returns:
        List of compressed strings like "18:1-3" or "0-31:1-0..2"

    Steps:
      1) Sort by (Dev, A, B).
      2) For each (Dev, A), compress contiguous B's into B-start..B-end.
      3) For each (A, B-start..B-end), if multiple contiguous Devs share
         that exact (A,B-range), compress into DevStart-DevEnd:A-Bstart..Bend.
      4) Return list of strings.
    """
    if not dev_loc_keys:
        return []

    dev_loc_keys_sorted = sorted(dev_loc_keys)  # (Dev, A, B)

    # Step 2: within Dev,A, merge B ranges
    per_dev_a_ranges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for dev, a, b in dev_loc_keys_sorted:
        key = (dev, a)
        ranges = per_dev_a_ranges.setdefault(key, [])
        if not ranges:
            ranges.append((b, b))
        else:
            start, end = ranges[-1]
            if b == end + 1:
                ranges[-1] = (start, b)
            else:
                ranges.append((b, b))

    # Step 3: group by (A, Bstart, Bend) to merge device ranges
    by_range: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for (dev, a), ranges in per_dev_a_ranges.items():
        for b_start, b_end in ranges:
            by_range[(a, b_start, b_end)].append(dev)

    compressed: list[str] = []
    for (a, b_start, b_end), devs in sorted(by_range.items()):
        devs_sorted = sorted(devs)

        run_start = devs_sorted[0]
        prev = run_start
        for d in devs_sorted[1:]:
            if d == prev + 1:
                prev = d
            else:
                # flush previous dev run
                if run_start == prev:
                    dev_str = f"{run_start}"
                else:
                    dev_str = f"{run_start}-{prev}"
                if b_start == b_end:
                    compressed.append(f"{dev_str}:{a}-{b_start}")
                else:
                    compressed.append(f"{dev_str}:{a}-{b_start}..{b_end}")
                run_start = d
                prev = d

        # flush final dev run
        if run_start == prev:
            dev_str = f"{run_start}"
        else:
            dev_str = f"{run_start}-{prev}"
        if b_start == b_end:
            compressed.append(f"{dev_str}:{a}-{b_start}")
        else:
            compressed.append(f"{dev_str}:{a}-{b_start}..{b_end}")

    return compressed


def _aggregate_callstacks_v3(per_core_results) -> list[AggregatedCallstackRow]:
    """
    Aggregate per-core callstack results into groups by canonical callstack.

    Returns a list of AggregatedCallstackRow objects, sorted by descending count.
    """
    if not per_core_results:
        return []

    groups: dict[tuple, dict] = {}

    for check in per_core_results:
        data: CallstacksData = check.result
        if data is None:
            continue

        kcwm = data.kernel_callstack_with_message
        key = _canonical_callstack_key(kcwm)

        if key not in groups:
            # Extract kernel name from dispatcher core data
            kernel_name = getattr(data.dispatcher_core_data, "kernel_name", None)
            groups[key] = {
                "kcwm": kcwm,
                "kernel_name": kernel_name,
                "count": 0,
                "risc_names": set(),
                "dev_loc_keys": [],
            }

        g = groups[key]
        g["count"] += 1
        g["risc_names"].add(check.risc_name)

        dev_id = check.device_description.device.id
        x, y = check.location.to("noc0")
        g["dev_loc_keys"].append((dev_id, x, y))

    rows: list[AggregatedCallstackRow] = []
    for g in groups.values():
        kcwm = g["kcwm"]
        formatted_callstack = format_callstack_with_message(kcwm)
        risc_names_str = ",".join(sorted(g["risc_names"]))
        compressed_locs = _compress_devices_locs_v3(g["dev_loc_keys"])

        rows.append(
            AggregatedCallstackRow(
                kernel_name=g["kernel_name"],
                callstack=formatted_callstack,
                num_callstacks=g["count"],
                risc_names=risc_names_str,
                devices_locs=compressed_locs,
            )
        )

    # Sort by descending count, then risc_names, then callstack
    rows.sort(key=lambda r: (-r.num_callstacks, r.risc_names, r.callstack))
    return rows


def dump_callstacks(
    location: OnChipCoordinate,
    risc_name: str,
    callstack_provider: CallstackProvider,
    show_all_cores: bool = False,
) -> CallstacksData | None:
    try:
        # Skip DONE cores unless --all-cores is specified
        if not show_all_cores:
            dispatcher_core_data = callstack_provider.dispatcher_data.get_cached_core_data(location, risc_name)
            if dispatcher_core_data.go_message == "DONE":
                return None
        return callstack_provider.get_callstacks(location, risc_name)
    except TimeoutDeviceRegisterError:
        raise
    except Exception as e:
        log_check_risc(
            risc_name,
            location,
            False,
            f"[warning]Failed to dump callstacks: {e}[/]",
        )
        return None


def run(args, context: Context):
    show_all_cores: bool = args["--all-cores"]
    per_core: bool = args["--per-core"]
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]
    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)
    per_core_results = run_checks.run_per_core_check(
        lambda location, risc_name: dump_callstacks(
            location,
            risc_name,
            callstack_provider,
            show_all_cores,
        ),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    # If --per-core is specified or no results, return per-core view
    if per_core or not per_core_results:
        return per_core_results

    # Default: return aggregated view
    return _aggregate_callstacks_v3(per_core_results)


if __name__ == "__main__":
    run_script()
