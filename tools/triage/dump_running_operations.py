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
    Summarizes currently running operations across all inspected cores.

    Output:
      - One row per unique **Op Id** currently observed on at least one inspected core
        (Op Id == dispatcher host_assigned_id).
      - Includes current Op Id/Name/Params, previous Op (Prev Op Id/Name/Params), and device/core coverage:
          - Device Cnt / Core Cnt: total unique devices/cores running this Op Id
          - Devices / Cores: enumerated lists (may be truncated with "..." for readability)

    Data sources:
      - Dispatcher mailboxes (`dispatcher_data.py`):
          - Provides Op Id (host_assigned_id) per core, plus the previous launch entry’s Op Id.
      - Inspector mesh workloads (`operation_runtime_map.py`, derived from `inspector_data.py`):
          - Provides mapping: runtime_id -> (operation name, operation parameters).

    Key assumptions / caveats:
      - Op Id (dispatcher host_assigned_id) is expected to match Inspector runtime_id.

    Heuristic:
      - The smallest Op Id among currently-running cores can sometimes indicate
        an older operation that has not completed.

    Caveat:
      - In replay / cached execution modes, Op Id may not monotonically increase,
        so this heuristic can be wrong.

Owner:
    miacim
"""

from dataclasses import dataclass

from dispatcher_data import run as get_dispatcher_data, DispatcherData, DispatcherCoreData
from operation_runtime_map import run as get_operation_runtime_map
from operation_param_parser import ParameterParser
from run_checks import (
    run as get_run_checks,
    RunChecks,
    device_description_serializer,
)
from triage import (
    ScriptConfig,
    collection_serializer,
    hex_serializer,
    log_check,
    log_check_risc,
    triage_field,
    run_script,
)
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.elf import ElfVariable
from ttexalens.umd_device import TimeoutDeviceRegisterError

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "operation_runtime_map"],
)

# Core filtering
BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]

# Display limits
MAX_CORES_DISPLAYED = 5  # Maximum cores shown per operation
MAX_DEVICES_DISPLAYED = 5  # Maximum devices shown per operation


# ============================================================================
# Running Operation Data Structures
# ============================================================================


@dataclass
class RunningOperationSummary:
    """Summary of a running operation across all cores executing it."""

    host_assigned_id: int = triage_field("Op Id")
    operation_name: str = triage_field("Op Name")
    operation_parameters: str = triage_field("Op Params")
    previous_host_assigned_id: int | None = triage_field("Prev Op Id")
    previous_operation_name: str = triage_field("Prev Op Name")
    previous_operation_parameters: str = triage_field("Prev Op Params")
    device_count: int = triage_field("Device Cnt")
    core_count: int = triage_field("Core Cnt")
    devices: list[str] = triage_field("Devices", collection_serializer(", "))
    cores: list[str] = triage_field("Cores", collection_serializer("\n"))


class RunningOperationAggregation:
    """Mutable accumulator for all cores running operations under the same host assigned ID."""

    def __init__(
        self,
        host_assigned_id: int,
        operation_name: str = "",
        operation_parameters: str = "",
        previous_host_assigned_id: int | None = None,
        previous_operation_name: str = "",
        previous_operation_parameters: str = "",
    ):
        self.host_assigned_id = host_assigned_id
        self.operation_name = operation_name
        self.operation_parameters = operation_parameters
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
    ):
        """Add a core to this operation aggregation."""
        self.core_locations.add(_format_core_location(device_label, location))
        self.device_labels.add(device_label)

    def to_summary(self) -> RunningOperationSummary:
        """Convert aggregation to immutable summary with limited display."""
        devices = sorted(self.device_labels)
        devices_to_display = (
            devices if len(devices) <= MAX_DEVICES_DISPLAYED else devices[:MAX_DEVICES_DISPLAYED] + ["..."]
        )

        unique_cores = sorted(self.core_locations)
        cores_to_display = (
            unique_cores if len(unique_cores) <= MAX_CORES_DISPLAYED else unique_cores[:MAX_CORES_DISPLAYED] + ["..."]
        )

        # Format operation parameters for display
        # If parameters are available, parse them; otherwise show N/A
        operation_params_display = "N/A"
        if self.operation_parameters:
            operation_params_display = ParameterParser.format_multiline(self.operation_parameters)

        prev_operation_params_display = "N/A"
        if self.previous_operation_parameters:
            prev_operation_params_display = ParameterParser.format_multiline(self.previous_operation_parameters)

        return RunningOperationSummary(
            host_assigned_id=self.host_assigned_id,
            operation_name=self.operation_name or "N/A",
            operation_parameters=operation_params_display,
            previous_host_assigned_id=self.previous_host_assigned_id,
            previous_operation_name=self.previous_operation_name or "N/A",
            previous_operation_parameters=prev_operation_params_display,
            device_count=len(self.device_labels),
            core_count=len(self.core_locations),
            devices=devices_to_display,
            cores=cores_to_display,
        )


# ============================================================================
# Script Logic - Data Collection and Aggregation
# ============================================================================


def _format_core_location(device_label: str | None, location: OnChipCoordinate | None) -> str:
    """Format a core location as device:coordinate string."""
    if location is None:
        return "N/A"
    if device_label is None:
        return location.to_str("noc0")
    return f"{device_label}:{location.to_str('noc0')}"


def _collect_dispatcher_data(
    dispatcher_data: DispatcherData, location: OnChipCoordinate, risc_name: str, show_all_cores: bool = False
) -> DispatcherCoreData | None:
    """Collect dispatcher data for a single core.

    Args:
        dispatcher_data: Dispatcher data cache
        location: Core location
        risc_name: RISC-V core name
        show_all_cores: If False, filter out DONE cores

    Returns:
        DispatcherCoreData if relevant, None otherwise
    """
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

    # Filter out DONE cores unless explicitly requested
    if not show_all_cores and dispatcher_core_data.go_message == "DONE":
        return None

    # Skip cores with no host assigned ID or ID of 0
    host_assigned_id = dispatcher_core_data.host_assigned_id
    if host_assigned_id in (None, 0):
        return None

    return dispatcher_core_data


def _collect_running_operations(
    dispatcher_data: DispatcherData,
    run_checks: RunChecks,
    runtime_id_to_operation: dict[int, tuple[str, str]],
    show_all_cores: bool = False,
) -> list[RunningOperationSummary] | None:
    """Collect and aggregate running operations across all cores.

    Args:
        dispatcher_data: Dispatcher data cache
        run_checks: Run checks infrastructure for core iteration
        runtime_id_to_operation: Mapping from runtime_id -> (operation_name, operation_parameters)
        show_all_cores: If False, filter out DONE cores

    Returns:
        List of operation summaries, one per unique host assigned ID
    """
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

    try:
        # Process results and aggregate by host_assigned_id
        for check_result in collected_results:
            if check_result.result is None:
                continue

            if not isinstance(check_result.result, DispatcherCoreData):
                continue

            dispatcher_core_data: DispatcherCoreData = check_result.result

            if dispatcher_core_data.host_assigned_id in (None, 0):
                continue

            # Get operation name and parameters from runtime_id mapping
            # The host_assigned_id from dispatcher should match runtime_id from inspector
            operation_name, operation_parameters = runtime_id_to_operation.get(
                dispatcher_core_data.host_assigned_id, ("", "")
            )

            prev_runtime_id = dispatcher_core_data.previous_host_assigned_id

            if prev_runtime_id not in (None, 0):
                prev_operation_name, prev_operation_parameters = runtime_id_to_operation.get(prev_runtime_id, ("", ""))
            else:
                prev_operation_name, prev_operation_parameters = "", ""

            # Get or create aggregation for this host_assigned_id
            aggregation = aggregations.setdefault(
                dispatcher_core_data.host_assigned_id,
                RunningOperationAggregation(
                    dispatcher_core_data.host_assigned_id,
                    operation_name,
                    operation_parameters,
                    prev_runtime_id if prev_runtime_id > 0 else None,
                    prev_operation_name,
                    prev_operation_parameters,
                ),
            )

            # Add this core to the aggregation
            aggregation.add_core(
                device_description_serializer(check_result.device_description),
                check_result.location,
                check_result.risc_name,
                dispatcher_core_data,
            )
    except Exception as e:
        log_check(False, f"Failed to collect running operations: {e}")
        return None

    if not aggregations:
        return None

    # Convert aggregations to summaries, sorted by host_assigned_id
    return [aggregations[host_assigned_id].to_summary() for host_assigned_id in sorted(aggregations.keys())]


# ============================================================================
# Entry Point
# ============================================================================


def run(args, context: Context):
    """Main entry point for the script."""
    show_all_cores: bool = args["--include-done"]
    dispatcher_data = get_dispatcher_data(args, context)
    run_checks = get_run_checks(args, context)
    runtime_id_to_operation = get_operation_runtime_map(args, context)
    return _collect_running_operations(dispatcher_data, run_checks, runtime_id_to_operation, show_all_cores)


if __name__ == "__main__":
    run_script()
