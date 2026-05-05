#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    running_ops_aggregation [--include-done]

Options:
    --include-done   Show all cores including ones with Go Message = DONE.
                     By default, DONE cores are filtered out.

Description:
    Data provider that scans every dispatcher core, filters DONE/idle cores
    (unless --include-done is set), looks up each running mailbox host_assigned_id
    in the Inspector runtime map, and produces a `RunningOpsAggregation` (the
    per-host_assigned_id `RunningOperationAggregation` dict + the runtime map).

    Cached with @triage_singleton so dump_running_operations, dump_op_window,
    and dump_galaxy_mesh share a single iteration over the dispatcher cores.

Owner:
    miacim
"""

from __future__ import annotations

from dataclasses import dataclass

from dispatcher_data import run as get_dispatcher_data, DispatcherData, DispatcherCoreData
from operation_runtime_map import (
    run as get_operation_runtime_map,
    OperationRuntimeMap,
    OperationInfo,
)
from run_checks import (
    run as get_run_checks,
    RunChecks,
    device_description_serializer,
)
from triage import (
    ScriptConfig,
    log_check,
    log_check_risc,
    triage_singleton,
    run_script,
)
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.umd_device import TimeoutDeviceRegisterError


script_config = ScriptConfig(
    data_provider=True,
    depends=["run_checks", "dispatcher_data", "operation_runtime_map"],
)


# Core filtering — matches the historical block list from dump_running_operations.
BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth", "dram"]


def _format_core_location(device_label: str | None, location: OnChipCoordinate | None) -> str:
    """Format a core location as device:coordinate string."""
    if location is None:
        return "N/A"
    if device_label is None:
        return location.to_user_str()
    return f"{device_label}:{location.to_user_str()}"


class RunningOperationAggregation:
    """Mutable accumulator for all cores running operations under the same host assigned ID."""

    def __init__(
        self,
        host_assigned_id: int,
        operation_name: str = "",
        operation_parameters: str = "",
        trace_id: int | None = None,
        previous_host_assigned_id: int | None = None,
        previous_operation_name: str = "",
        previous_operation_parameters: str = "",
    ):
        self.host_assigned_id = host_assigned_id
        self.operation_name = operation_name
        self.operation_parameters = operation_parameters
        self.trace_id = trace_id
        self.previous_host_assigned_id = previous_host_assigned_id
        self.previous_operation_name = previous_operation_name
        self.previous_operation_parameters = previous_operation_parameters
        self.core_locations: set[str] = set()
        self.device_labels: set[str] = set()

    def add_core(
        self,
        device_label: str,
        location: OnChipCoordinate,
        risc_name: str,
        dispatcher_core_data: DispatcherCoreData,
    ) -> None:
        self.core_locations.add(_format_core_location(device_label, location))
        self.device_labels.add(device_label)


@dataclass
class RunningOpsAggregation:
    """Bundle of aggregations + runtime map shared by all running-ops consumers."""

    aggregations: dict[int, "RunningOperationAggregation"]
    runtime_id_to_operation: OperationRuntimeMap


def _collect_dispatcher_data(
    dispatcher_data: DispatcherData,
    location: OnChipCoordinate,
    risc_name: str,
    show_all_cores: bool,
) -> DispatcherCoreData | None:
    if not dispatcher_data.risc_enabled(risc_name):
        return None

    try:
        dispatcher_core_data = dispatcher_data.get_cached_core_data(location, risc_name)
    except TimeoutDeviceRegisterError:
        raise
    except Exception as e:
        log_check_risc(
            risc_name,
            location,
            False,
            f"Failed to read dispatcher data for running operations aggregation: {e}",
        )
        return None

    if not show_all_cores and dispatcher_core_data.go_message == "DONE":
        return None

    if dispatcher_core_data.host_assigned_id in (None, 0):
        return None

    return dispatcher_core_data


@triage_singleton
def run(args, context: Context) -> RunningOpsAggregation:
    """Build the running-ops aggregation. Cached per (args, context)."""
    show_all_cores: bool = args["--include-done"]

    dispatcher_data = get_dispatcher_data(args, context)
    run_checks: RunChecks = get_run_checks(args, context)
    runtime_id_to_operation = get_operation_runtime_map(args, context)

    collected_results = run_checks.run_per_core_check(
        lambda location, risc_name: _collect_dispatcher_data(dispatcher_data, location, risc_name, show_all_cores),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    aggregations: dict[int, RunningOperationAggregation] = {}

    if not collected_results:
        return RunningOpsAggregation(aggregations, runtime_id_to_operation)

    # Per-iteration error isolation: a single bad row should not drop the whole table.
    for check_result in collected_results:
        try:
            if check_result.result is None:
                continue
            if not isinstance(check_result.result, DispatcherCoreData):
                continue

            dispatcher_core_data: DispatcherCoreData = check_result.result
            if dispatcher_core_data.host_assigned_id in (None, 0):
                continue

            dispatch_mode = dispatcher_core_data.dispatch_mode

            op_info = (
                runtime_id_to_operation.lookup(dispatcher_core_data.host_assigned_id, dispatch_mode)
                or OperationInfo.empty()
            )

            if dispatch_mode == "HOST":
                prev_runtime_id = None
                prev_op_info = OperationInfo.empty()
            else:
                prev_runtime_id = dispatcher_core_data.previous_host_assigned_id
                if prev_runtime_id not in (None, 0):
                    prev_op_info = (
                        runtime_id_to_operation.lookup(prev_runtime_id, dispatch_mode) or OperationInfo.empty()
                    )
                else:
                    prev_op_info = OperationInfo.empty()

            aggregation = aggregations.setdefault(
                dispatcher_core_data.host_assigned_id,
                RunningOperationAggregation(
                    dispatcher_core_data.host_assigned_id,
                    op_info.name,
                    op_info.parameters,
                    op_info.trace_id,
                    prev_runtime_id if prev_runtime_id and prev_runtime_id > 0 else None,
                    prev_op_info.name,
                    prev_op_info.parameters,
                ),
            )

            aggregation.add_core(
                device_description_serializer(check_result.device_description),
                check_result.location,
                check_result.risc_name,
                dispatcher_core_data,
            )
        except Exception as e:
            log_check(False, f"Failed to aggregate one core's running-op data: {e}")
            continue

    return RunningOpsAggregation(aggregations, runtime_id_to_operation)


if __name__ == "__main__":
    run_script()
