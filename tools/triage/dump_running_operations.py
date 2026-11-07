#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_running_operations [--full-cores]

Options:
    --full-cores     Print the full list of cores per operation instead of truncating with "...".
                     Devices are always shown in full.

Description:
    Summarizes currently running operations across all inspected cores.

    Output:
      - One row per unique **Op Id** currently observed on at least one inspected core
        (Op Id == dispatcher host_assigned_id).
      - Includes current Op Id/Name/Params, previous Op (Prev Op Id/Name/Params), and device/core coverage:
          - Device Cnt / Core Cnt: total unique devices/cores running this Op Id
          - Devices / Cores: enumerated lists (may be truncated with "..." for readability)
          - CT Args / Common RT Args: one line per kernel the op runs, with its compile-time args and
            its common runtime args (see kernel_args_provider)

    Companion scripts:
      - `dump_op_window` - context table of ops with id near the currently-running set.
      - `dump_op_mesh` - 2D mesh-grid view of running ops, sourced from Inspector's system mesh.
      All share the same aggregation via `operation_provider` (use `--include-done`
      on the triage CLI to surface DONE cores in every consumer).

    Data sources:
      - `operation_provider` (per-core dispatcher data + Inspector runtime map).

Owner:
    onenezicTT
"""

from dataclasses import dataclass

from kernel_args_provider import run as get_kernel_args_provider, KernelArgsProvider, serialize_kernel_args
from operation_param_parser import ParameterParser
from operation_provider import (
    run as get_operation_provider,
    RunningOperationAggregation,
)
from triage import (
    ScriptConfig,
    ScriptPriority,
    collection_serializer,
    triage_field,
    run_script,
)
from ttexalens.context import Context


script_config = ScriptConfig(
    depends=["operation_provider", "kernel_args_provider"],
    priority=ScriptPriority.HIGH,
)


MAX_CORES_DISPLAYED = 5


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
    compile_time_args: list[str] = triage_field("CT Args", collection_serializer("\n"))
    common_runtime_args: list[str] = triage_field("Common RT Args", collection_serializer("\n"))


def _to_summary(
    agg: RunningOperationAggregation, full_cores: bool, kernel_args_provider: KernelArgsProvider
) -> RunningOperationSummary:
    devices = sorted(agg.device_labels)

    compile_time_args = []
    common_runtime_args = []
    for _, kernel in sorted(agg.kernels.items()):
        kernel_args = kernel_args_provider.get_cached_kernel_args(kernel.risc_location)
        compile_time_args.append(f"{kernel.name}: {serialize_kernel_args(kernel_args.compile_time_args)}")
        common_runtime_args.append(f"{kernel.name}: {serialize_kernel_args(kernel_args.common_runtime_args)}")

    unique_cores = sorted(agg.core_locations)
    cores_to_display = (
        unique_cores
        if full_cores or len(unique_cores) <= MAX_CORES_DISPLAYED
        else unique_cores[:MAX_CORES_DISPLAYED] + ["..."]
    )

    operation_params_display = (
        ParameterParser.format_multiline(agg.operation_parameters) if agg.operation_parameters else "N/A"
    )
    prev_operation_params_display = (
        ParameterParser.format_multiline(agg.previous_operation_parameters)
        if agg.previous_operation_parameters
        else "N/A"
    )

    # Flag replayed ops inline with the name so the table makes it
    # obvious which dispatches are coming from trace replay.
    display_name = agg.operation_name or "N/A"
    if agg.operation_name and agg.trace_id is not None:
        display_name = f"{display_name} (trace id: {agg.trace_id})"

    return RunningOperationSummary(
        host_assigned_id=agg.host_assigned_id,
        operation_name=display_name,
        operation_parameters=operation_params_display,
        previous_host_assigned_id=agg.previous_host_assigned_id,
        previous_operation_name=agg.previous_operation_name or "N/A",
        previous_operation_parameters=prev_operation_params_display,
        device_count=len(agg.device_labels),
        core_count=len(agg.core_locations),
        devices=devices,
        cores=cores_to_display,
        compile_time_args=compile_time_args,
        common_runtime_args=common_runtime_args,
    )


def run(args, context: Context):
    """Pure renderer: read aggregations from the data provider, emit summaries."""
    full_cores: bool = args["--full-cores"]

    bundle = get_operation_provider(args, context)
    if not bundle.aggregations:
        return None

    kernel_args_provider = get_kernel_args_provider(args, context)
    return [
        _to_summary(bundle.aggregations[hid], full_cores, kernel_args_provider)
        for hid in sorted(bundle.aggregations.keys())
    ]


if __name__ == "__main__":
    run_script()
