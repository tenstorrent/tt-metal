#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_aggregated_callstacks [--all-cores]

Options:
    --all-cores        Show all cores including ones with Go Message = DONE. By default, DONE cores are filtered out.

Description:
    Aggregates callstacks by (Kernel Id, normalized PC, Op Id) and shows:
      - Kernel Id / Kernel Name
      - Op Id (host_assigned_id, for correlation with dump_running_operations.py)
      - Callstack
      - # of Cores
      - RISC Names
      - Locations (device:core)
    This significantly reduces the number of rows vs raw dump_callstacks.

    Note: This script is disabled by default. To enable it, set the environment variable:
        TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS=1

Owner:
    onenezic
"""

import os
from dataclasses import dataclass

from triage import ScriptConfig, run_script, triage_field, collection_serializer
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

script_config = ScriptConfig(
    depends=["run_checks", "callstack_provider"],
    disabled=os.environ.get("TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS") != "1",
)

BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]


@dataclass
class AggregatedCallstackRow:
    kernel_id: int | None = triage_field("Kernel Id")
    kernel_name: str | None = triage_field("Kernel Name")
    op_id: int | str | None = triage_field("Op Id")  # host_assigned_id from dispatcher, or "N/A (read error)"
    callstack: KernelCallstackWithMessage = triage_field("Kernel Callstack", format_callstack_with_message)
    core_count: int = triage_field("# of Cores")
    risc_names: list[str] = triage_field("RISC Names", collection_serializer(", "))
    locations: list[str] = triage_field("Locations (device:core)", collection_serializer("\n"))


class AggregationBucket:
    """Helper class that accumulates per-group data for aggregation."""

    def __init__(self, first: CallstacksData):
        d = first.dispatcher_core_data
        self.kernel_id = d.watcher_kernel_id
        self.kernel_name = d.kernel_name
        # op_id will be set later with error handling
        self.op_id = None
        self.callstack = first.kernel_callstack_with_message
        self.core_locations: set[str] = set()
        self.riscs: set[str] = set()

    def add_core(self, device_label: str, location: OnChipCoordinate, risc_name: str):
        """Add a core to this aggregation bucket."""
        self.core_locations.add(f"{device_label}:{location.to_str('noc0')}")
        self.riscs.add(risc_name)

    def to_row(self) -> AggregatedCallstackRow:
        """Convert the bucket to an immutable row for display."""
        return AggregatedCallstackRow(
            kernel_id=self.kernel_id,
            kernel_name=self.kernel_name,
            op_id=self.op_id,
            callstack=self.callstack,
            core_count=len(self.core_locations),
            risc_names=sorted(self.riscs),
            locations=sorted(self.core_locations),
        )


def _collect_aggregated(
    callstack_provider: CallstackProvider,
    run_checks,
    show_all_cores: bool,
) -> list[AggregatedCallstackRow] | None:
    """Collect callstacks and aggregate by (kernel_id, normalized_pc, op_id)."""

    def per_core(location: OnChipCoordinate, risc_name: str) -> CallstacksData | None:
        # Filter DONE cores, like dump_callstacks.py does
        if not show_all_cores:
            d = callstack_provider.dispatcher_data.get_cached_core_data(location, risc_name)
            if d.go_message == "DONE":
                return None

        # This will use the new caching in CallstackProvider
        return callstack_provider.get_callstacks(location, risc_name)

    results = run_checks.run_per_core_check(
        per_core,
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    if not results:
        return None

    # Aggregate by (kernel_id, normalized_pc, op_id)
    buckets: dict[tuple[int | None, int | None, int | None], AggregationBucket] = {}

    for check_result in results:
        if check_result.result is None:
            continue

        try:
            cs_data: CallstacksData = check_result.result
            d = cs_data.dispatcher_core_data

            op_id = d.host_assigned_id
            pc = cs_data.pc

            # Normalize PC into kernel space when kernel_offset is available
            if d.kernel_offset is not None and pc is not None:
                normalized_pc = pc - d.kernel_offset
            else:
                normalized_pc = pc

            key = (d.watcher_kernel_id, normalized_pc, op_id)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = AggregationBucket(cs_data)
                # Store the safe op_id value
                bucket.op_id = op_id
                buckets[key] = bucket

            # Get device label from device description
            device_label = device_description_serializer(check_result.device_description)
            bucket.add_core(device_label, check_result.location, check_result.risc_name)

        except Exception as e:
            # If ANY error occurs processing this core, skip it and continue
            # This prevents one bad core from crashing the entire aggregation
            continue

    # Sort descending by # of cores
    sorted_buckets = sorted(buckets.values(), key=lambda b: len(b.core_locations), reverse=True)
    return [b.to_row() for b in sorted_buckets]


def run(args, context: Context):
    """Main entry point for the script."""
    show_all_cores: bool = args["--all-cores"]
    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)
    return _collect_aggregated(callstack_provider, run_checks, show_all_cores)


if __name__ == "__main__":
    run_script()
