#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    operation_provider [--include-done]

Options:
    --include-done   Show all cores including ones with Go Message = DONE.
                     By default, DONE cores are filtered out.

Description:
    Data provider for the running-ops view chain. Builds two things:

      1. An `OperationRuntimeMap` - Inspector runtime_id → OperationInfo
         (name, parameters, trace_id). The mailbox holds the raw runtime_id
         for both fast and slow dispatch (see tt_metal/impl/program/dispatch.cpp
         and tt_metal/impl/host_api/tt_metal.cpp).
      2. A per-`host_assigned_id` aggregation: scans every dispatcher core,
         filters DONE/idle cores (unless --include-done), looks each running
         host_assigned_id up in the runtime map, and accumulates per-op
         device/core sets.

    Cached with @triage_singleton so dump_running_operations, dump_op_window,
    and dump_op_mesh share a single iteration over the dispatcher cores.

Owner:
    onenezicTT
"""

from __future__ import annotations

from dataclasses import dataclass

from dispatcher_data import run as get_dispatcher_data, DispatcherData, DispatcherCoreData
from inspector_data import run as get_inspector_data, InspectorData
from run_checks import (
    run as get_run_checks,
    RunChecks,
    device_description_serializer,
)
from triage import (
    ScriptConfig,
    ScriptPriority,
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
    depends=["run_checks", "dispatcher_data", "inspector_data"],
    priority=ScriptPriority.HIGH,
)


# Core filtering - matches the historical block list from dump_running_operations.
BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth", "dram"]

# Matches the sentinel in rpc.capnp (kNoTraceId). Capnp has no null for primitives.
_NO_TRACE_ID = 0xFFFFFFFF


@dataclass
class OperationInfo:
    name: str
    parameters: str
    trace_id: int | None

    @staticmethod
    def empty() -> "OperationInfo":
        return OperationInfo(name="", parameters="", trace_id=None)


class OperationRuntimeMap:
    """Resolves dispatcher `host_assigned_id` values to Inspector-recorded op info."""

    def __init__(self, runtime_id_map: dict[int, OperationInfo]):
        self._map = runtime_id_map

    def __len__(self) -> int:
        return len(self._map)

    def items(self):
        return self._map.items()

    def lookup(self, host_assigned_id: int) -> OperationInfo | None:
        """Resolve host_assigned_id to its Inspector entry, or None if not found."""
        return self._map.get(host_assigned_id)


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

    def add_core(self, device_label: str, location: OnChipCoordinate) -> None:
        self.core_locations.add(_format_core_location(device_label, location))
        self.device_labels.add(device_label)


@dataclass
class RunningOpsAggregation:
    """Bundle of aggregations + runtime map shared by all running-ops consumers."""

    aggregations: dict[int, "RunningOperationAggregation"]
    runtime_id_to_operation: OperationRuntimeMap


def _build_runtime_id_map(inspector_data: InspectorData) -> OperationRuntimeMap:
    runtime_id_map: dict[int, OperationInfo] = {}
    try:
        runtime_entries_result = inspector_data.getMeshWorkloadRuntimeEntries()
        for entry in runtime_entries_result.runtimeEntries:
            raw_trace_id = int(entry.traceId)
            trace_id = None if raw_trace_id == _NO_TRACE_ID else raw_trace_id
            runtime_id_map[int(entry.runtimeId)] = OperationInfo(
                name=entry.operationName,
                parameters=entry.operationParameters,
                trace_id=trace_id,
            )
        log_check(True, f"Built runtime_id map with {len(runtime_id_map)} operation(s)")
    except Exception as e:
        log_check(False, f"Failed to build runtime_id to operation map: {e}")
        log_check(False, "Operation names and parameters will not be available")
    return OperationRuntimeMap(runtime_id_map)


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
    """Build the running-ops aggregation + the runtime map. Cached per (args, context)."""
    show_all_cores: bool = args["--include-done"]

    inspector_data = get_inspector_data(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    run_checks: RunChecks = get_run_checks(args, context)

    runtime_id_to_operation = _build_runtime_id_map(inspector_data)

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
            host_assigned_id = dispatcher_core_data.host_assigned_id
            if host_assigned_id is None or host_assigned_id == 0:
                continue

            dispatch_mode = dispatcher_core_data.dispatch_mode

            op_info = runtime_id_to_operation.lookup(host_assigned_id) or OperationInfo.empty()

            # Slow dispatch (HOST) overwrites a single launch slot in the mailbox, so the
            # "previous" entry is stale/invalid there.
            if dispatch_mode == "HOST":
                prev_runtime_id = None
                prev_op_info = OperationInfo.empty()
            else:
                prev_runtime_id = dispatcher_core_data.previous_host_assigned_id
                if prev_runtime_id is not None and prev_runtime_id != 0:
                    prev_op_info = runtime_id_to_operation.lookup(prev_runtime_id) or OperationInfo.empty()
                else:
                    prev_op_info = OperationInfo.empty()

            aggregation = aggregations.setdefault(
                host_assigned_id,
                RunningOperationAggregation(
                    host_assigned_id,
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
            )
        except Exception as e:
            log_check(False, f"Failed to aggregate one core's running-op data: {e}")
            continue

    return RunningOpsAggregation(aggregations, runtime_id_to_operation)


if __name__ == "__main__":
    run_script()
