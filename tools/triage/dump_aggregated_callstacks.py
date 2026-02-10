#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_aggregated_callstacks [--all-cores] [--device-visualization]

Options:
    --all-cores                      Show all cores including ones with Go Message = DONE. By default, DONE cores are filtered out.
    --device-visualization           Show device visualizations instead of plain coordinate lists in the Locations column.
Description:
    Aggregates callstacks by (Kernel Id, normalized PC, Op Id) and shows:
      - Kernel Id / Kernel Name
      - Op Id (host_assigned_id, for correlation with dump_running_operations.py)
      - Callstack
      - # of Cores
      - RISC Names
      - Locations or device visualizations (if --device-visualization)
    This significantly reduces the number of rows vs raw dump_callstacks.

    When --device-visualization is specified, the last column ("Locations")
    will show per-device ASCII visualizations (device grid)
    highlighting only the cores that belong to this aggregated row.

    Note: This script is disabled by default. To enable it, set the environment variable:
        TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS=1

Owner:
    onenezicTT
"""

import os
from collections import defaultdict
from dataclasses import dataclass

from triage import ScriptConfig, log_check_risc, run_script, triage_field, collection_serializer
from callstack_provider import (
    KernelCallstackWithMessage,
    format_callstack_with_message,
    CallstackProvider,
    CallstacksData,
    run as get_callstack_provider,
)
from run_checks import run as get_run_checks, device_description_serializer
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.umd_device import TimeoutDeviceRegisterError

script_config = ScriptConfig(
    depends=["run_checks", "callstack_provider"],
    disabled=os.environ.get("TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS") != "1",
)

BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]


def _render_device_for_bucket(
    device: Device,
    hits_for_device: dict[tuple[int, int], set[str]],
) -> str:
    """Render a compact device grid showing which cores are active."""

    header = f"Device {device.id}:"

    # Find all functional workers and determine max dimensions
    functional_workers = set()
    max_x, max_y = 0, 0
    locs_to_check = device.get_block_locations("functional_workers")
    locs_to_check.extend(device.get_block_locations("eth"))
    for block_loc in locs_to_check:
        x, y = block_loc._noc0_coord
        functional_workers.add((x, y))
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    lines = [header]

    col_header = "  " + "".join(f"{x:2}" for x in range(0, max_x + 1))
    lines.append(col_header)

    # Grid rows starting from y=0
    for y in range(0, max_y + 1):
        row = f"{y:2}"  # Row label (2 chars)
        for x in range(0, max_x + 1):
            if (x, y) in hits_for_device:
                row += " R"  # Active core (2 chars: space + R)
            elif (x, y) in functional_workers:
                row += " ."  # Inactive functional worker (2 chars: space + dot)
            else:
                row += "  "  # Not a functional worker (2 spaces)
        lines.append(row)

    return "\n".join(lines)


@dataclass
class AggregatedCallstackRow:
    kernel_id: int | None = triage_field("Kernel Id")
    kernel_name: str | None = triage_field("Kernel Name")
    op_id: int | str | None = triage_field("Op Id")  # host_assigned_id from dispatcher, or "N/A (read error)"
    callstack: KernelCallstackWithMessage = triage_field("Kernel Callstack", format_callstack_with_message)
    core_count: int = triage_field("# of Cores")
    risc_names: list[str] = triage_field("RISC Names", collection_serializer(", "))
    locations: list[str] = triage_field("Locations", collection_serializer("\n\n"))


class AggregationBucket:
    """Helper class that accumulates per-group data for aggregation."""

    def __init__(self, first: CallstacksData):
        d = first.dispatcher_core_data
        self.kernel_id = d.watcher_kernel_id
        self.kernel_name = d.kernel_name
        self.op_id = d.host_assigned_id
        self.callstack = first.kernel_callstack_with_message

        # Existing aggregation for plain text mode
        self.core_locations: set[str] = set()
        self.riscs: set[str] = set()

        # Helper for device visualization mode
        self.per_core_hits: dict[int, dict[tuple[int, int], set[str]]] = defaultdict(lambda: defaultdict(set))

    def add_core(self, location: OnChipCoordinate, risc_name: str, device_label: str | None):
        """Add a core to this aggregation bucket."""

        coord_str = location.to_str("noc0")
        if device_label:
            self.core_locations.add(f"{device_label}:{coord_str}")
        else:
            self.core_locations.add(coord_str)

        self.riscs.add(risc_name)

        dev_id = location._device.id
        x, y = location._noc0_coord
        self.per_core_hits[dev_id][(x, y)].add(risc_name)

    def to_row(self, visualize_devices: bool, context: Context) -> AggregatedCallstackRow:
        """Convert the bucket to an immutable row for display."""

        if visualize_devices:
            # Build one big visualization string for all devices in this bucket
            device_blocks: list[str] = []

            for dev_id in sorted(self.per_core_hits.keys()):
                device: Device = context.find_device_by_id(dev_id)
                hits_for_device = self.per_core_hits[dev_id]
                vis = _render_device_for_bucket(device, hits_for_device)
                device_blocks.append(vis)

            locations_list = [vis + "\n" for vis in device_blocks] if device_blocks else []
        else:
            locations_list = sorted(self.core_locations)

        return AggregatedCallstackRow(
            kernel_id=self.kernel_id,
            kernel_name=self.kernel_name,
            op_id=self.op_id,
            callstack=self.callstack,
            core_count=len(self.core_locations),
            risc_names=sorted(self.riscs),
            locations=locations_list,
        )


def _collect_aggregated(
    callstack_provider: CallstackProvider,
    run_checks,
    show_all_cores: bool,
    visualize_devices: bool,
    context: Context,
) -> list[AggregatedCallstackRow] | None:
    """Collect callstacks and aggregate by (kernel_id, normalized_pc, op_id)."""

    def per_core(location: OnChipCoordinate, risc_name: str) -> CallstacksData | None:
        try:
            # Filter DONE cores, like dump_callstacks.py does
            if not show_all_cores:
                d = callstack_provider.dispatcher_data.get_cached_core_data(location, risc_name)
                if d.go_message == "DONE":
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

    results = run_checks.run_per_core_check(
        per_core,
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    if not results:
        return None

    # Aggregate by (kernel_id, normalized_pc)
    buckets: dict[tuple[int | None, int | None], AggregationBucket] = {}

    for check_result in results:
        if check_result.result is None:
            continue

        cs_data: CallstacksData = check_result.result
        d = cs_data.dispatcher_core_data
        pc = cs_data.pc

        # Normalize PC into kernel space when kernel_offset is available
        if d.kernel_offset is not None and pc is not None:
            normalized_pc = pc - d.kernel_offset
        else:
            normalized_pc = pc

        key = (d.watcher_kernel_id, normalized_pc)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = AggregationBucket(cs_data)
            buckets[key] = bucket

        # Get device label from device description
        device_label = device_description_serializer(check_result.device_description)
        bucket.add_core(
            location=check_result.location,
            risc_name=check_result.risc_name,
            device_label=device_label,
        )

    # Sort descending by # of cores
    sorted_buckets = sorted(buckets.values(), key=lambda b: len(b.core_locations), reverse=True)
    return [b.to_row(visualize_devices, context) for b in sorted_buckets]


def run(args, context: Context):
    """Main entry point for the script."""
    show_all_cores: bool = args["--all-cores"]
    visualize_devices: bool = args["--device-visualization"]

    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)

    return _collect_aggregated(
        callstack_provider=callstack_provider,
        run_checks=run_checks,
        show_all_cores=show_all_cores,
        visualize_devices=visualize_devices,
        context=context,
    )


if __name__ == "__main__":
    run_script()
