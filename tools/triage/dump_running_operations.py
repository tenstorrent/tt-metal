#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_running_operations [--include-done]

Options:
    --include-done     Show all cores including ones with Go Message = DONE. By default, DONE cores are filtered out.

Description:
    Summarizes currently running operations across all inspected cores. Outputs one row per
    unique host assigned ID (excluding 0) together with the operations currently running,
    previously observed operations, device/core coverage, and the full list of cores executing
    each operation.
    By default, filters out cores with DONE status. Use --include-done to see all cores.
"""

from dataclasses import dataclass

from dispatcher_data import run as get_dispatcher_data, DispatcherData, DispatcherCoreData
from run_checks import (
    run as get_run_checks,
    RunChecks,
    DeviceDescription,
    device_description_serializer,
)
from triage import (
    ScriptConfig,
    collection_serializer,
    hex_serializer,
    log_check_risc,
    triage_field,
    run_script,
)
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data"],
)

BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]


@dataclass
class RunningOperationSummary:
    host_assigned_id: int = triage_field("Host Assigned ID", hex_serializer)
    operations: list[str] = triage_field("Operations", collection_serializer("\n"))
    previous_operations: list[str] = triage_field("Previous Operations", collection_serializer("\n"))
    devices: list[str] = triage_field("Device List", collection_serializer(", "))
    cores: list[str] = triage_field("Core List", collection_serializer("\n"))


MAX_CORES_DISPLAYED = 5
MAX_DEVICES_DISPLAYED = 5


class RunningOperationAggregation:
    """Mutable accumulator for all cores running operations under the same host assigned ID."""

    def __init__(self, host_assigned_id: int):
        self.host_assigned_id = host_assigned_id
        self.core_locations: set[str] = set()
        self.device_labels: set[str] = set()
        self.operations: set[str] = set()
        self.previous_operations: set[str] = set()

    def add_core(
        self,
        device_label: str,
        location: OnChipCoordinate,
        risc_name: str,
        dispatcher_core_data: DispatcherCoreData,
    ):
        self.core_locations.add(_format_core_location(device_label, location))
        self.device_labels.add(device_label)
        operation = _format_operation(dispatcher_core_data.kernel_name, dispatcher_core_data.watcher_kernel_id)
        if operation is not None:
            self.operations.add(operation)
        previous_operation = _format_operation(
            dispatcher_core_data.previous_kernel_name, dispatcher_core_data.watcher_previous_kernel_id
        )
        if previous_operation is not None:
            self.previous_operations.add(previous_operation)

    def to_summary(self) -> RunningOperationSummary:
        devices = sorted(self.device_labels)
        devices_to_display = (
            devices if len(devices) <= MAX_DEVICES_DISPLAYED else devices[:MAX_DEVICES_DISPLAYED] + ["..."]
        )
        operations = sorted(self.operations) if self.operations else ["N/A"]
        previous_operations = sorted(self.previous_operations) if self.previous_operations else ["N/A"]
        unique_cores = sorted(self.core_locations)
        cores_to_display = (
            unique_cores if len(unique_cores) <= MAX_CORES_DISPLAYED else unique_cores[:MAX_CORES_DISPLAYED] + ["..."]
        )
        return RunningOperationSummary(
            host_assigned_id=self.host_assigned_id,
            operations=operations,
            previous_operations=previous_operations,
            devices=devices_to_display,
            cores=cores_to_display,
        )


def _format_operation(kernel_name: str | None, watcher_kernel_id: int | None) -> str | None:
    if kernel_name:
        return f"Kernel: {kernel_name}"
    if watcher_kernel_id is not None and watcher_kernel_id >= 0:
        return f"Kernel ID: {watcher_kernel_id}"
    return None


def _format_core_location(device_label: str | None, location: OnChipCoordinate | None) -> str:
    if location is None:
        return "N/A"
    if device_label is None:
        return location.to_str("noc0")
    return f"{device_label}:{location.to_str('noc0')}"


def _collect_dispatcher_data(
    dispatcher_data: DispatcherData, location: OnChipCoordinate, risc_name: str, show_all_cores: bool = False
) -> DispatcherCoreData | None:
    try:
        dispatcher_core_data = dispatcher_data.get_cached_core_data(location, risc_name)
    except Exception as exc:
        log_check_risc(
            risc_name,
            location,
            False,
            f"Failed to read dispatcher data for running operations aggregation: {exc}",
        )
        return None
    if not show_all_cores and dispatcher_core_data.go_message == "DONE":
        return None

    host_assigned_id = dispatcher_core_data.host_assigned_id
    if host_assigned_id in (None, 0):
        return None
    return dispatcher_core_data


def _collect_running_operations(
    dispatcher_data: DispatcherData, run_checks: RunChecks, show_all_cores: bool = False
) -> list[RunningOperationSummary] | None:
    # Use run_checks infrastructure to iterate over all cores
    collected_results = run_checks.run_per_core_check(
        lambda location, risc_name: _collect_dispatcher_data(
            dispatcher_data,
            location,
            risc_name,
            show_all_cores,
        ),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    if not collected_results:
        return None

    aggregations: dict[int, RunningOperationAggregation] = {}

    # Process results
    for check_result in collected_results:
        if check_result.result is None:
            continue
        assert isinstance(check_result.result, DispatcherCoreData)
        dispatcher_core_data: DispatcherCoreData = check_result.result
        assert dispatcher_core_data.host_assigned_id is not None
        aggregation = aggregations.setdefault(
            dispatcher_core_data.host_assigned_id, RunningOperationAggregation(dispatcher_core_data.host_assigned_id)
        )
        aggregation.add_core(
            device_description_serializer(check_result.device_description),
            check_result.location,
            check_result.risc_name,
            dispatcher_core_data,
        )

    if not aggregations:
        return None

    return [aggregations[host_assigned_id].to_summary() for host_assigned_id in sorted(aggregations.keys())]


def run(args, context: Context):
    show_all_cores: bool = args["--include-done"]
    dispatcher_data = get_dispatcher_data(args, context)
    run_checks = get_run_checks(args, context)
    return _collect_running_operations(dispatcher_data, run_checks, show_all_cores)


if __name__ == "__main__":
    run_script()
