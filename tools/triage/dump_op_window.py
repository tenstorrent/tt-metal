#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_op_window [--op-window=<n>] [--full-devices]

Options:
    --op-window=<n>    Render ops with id in [min_running_op_id - n, max_running_op_id + n].
                       Use this to see what ran just before / what should have run next around
                       the currently-running set. [default: 10]
    --full-devices     Print the full list of devices per row instead of truncating with "...".

Description:
    Companion to `dump_running_operations`. Reads the same aggregation, then prints a
    context-window table sourced from the Inspector runtime map. Each row shows:
      - Op Id (raw runtime id)
      - Status (`RUNNING` if currently observed in dispatcher mailboxes, blank otherwise)
      - Op Name (Inspector)
      - Devices (set of device labels currently running this op id; blank for neighbors)

Owner:
    miacim
"""

from __future__ import annotations

from operation_runtime_map import OperationRuntimeMap, _decode_base_program_id
from running_ops_aggregation import (
    run as get_running_ops_aggregation,
    RunningOperationAggregation,
)
from triage import ScriptConfig, log_check, run_script
from ttexalens.context import Context


script_config = ScriptConfig(depends=["running_ops_aggregation"])


MAX_DEVICES_DISPLAYED = 5


def _resolve_running_raw_id(
    host_assigned_id: int,
    runtime_id_to_operation: OperationRuntimeMap,
) -> int | None:
    """Return the raw runtime_id matching a mailbox host_assigned_id, or None."""
    if runtime_id_to_operation.get_raw(host_assigned_id) is not None:
        return host_assigned_id
    decoded = _decode_base_program_id(host_assigned_id)
    if runtime_id_to_operation.get_raw(decoded) is not None:
        return decoded
    return None


def _build_raw_id_to_devices(
    aggregations: dict[int, RunningOperationAggregation],
    runtime_id_to_operation: OperationRuntimeMap,
) -> dict[int, set[str]]:
    """Map each running raw runtime_id to the set of device labels running it."""
    raw_to_devices: dict[int, set[str]] = {}
    for host_assigned_id, agg in aggregations.items():
        raw_id = _resolve_running_raw_id(host_assigned_id, runtime_id_to_operation)
        if raw_id is None:
            continue
        raw_to_devices.setdefault(raw_id, set()).update(agg.device_labels)
    return raw_to_devices


def _device_sort_key(device_label: str):
    """Sort numerically when possible, lexicographically otherwise."""
    try:
        return (0, int(device_label, 0))
    except (TypeError, ValueError):
        return (1, device_label)


def run(args, context: Context):
    try:
        window: int = int(args["--op-window"])
    except (TypeError, ValueError):
        window = 0
    full_devices: bool = args["--full-devices"]

    if window <= 0:
        log_check(False, f"Op-id window suppressed: --op-window={window} (must be > 0).")
        return None

    bundle = get_running_ops_aggregation(args, context)
    aggregations = bundle.aggregations
    runtime_id_to_operation = bundle.runtime_id_to_operation

    if not aggregations:
        log_check(False, "Op-id window suppressed: no running ops found in dispatcher mailboxes.")
        return None

    raw_to_devices = _build_raw_id_to_devices(aggregations, runtime_id_to_operation)
    running_raw_ids = set(raw_to_devices.keys())
    if not running_raw_ids:
        log_check(
            False,
            "Op-id window suppressed: no running host_assigned_id resolved to a known Inspector "
            "runtime id (Inspector data may be empty / out of sync).",
        )
        return None

    min_id = min(running_raw_ids) - window
    max_id = max(running_raw_ids) + window

    entries = sorted(
        (raw_id, op_info) for raw_id, op_info in runtime_id_to_operation.items() if min_id <= raw_id <= max_id
    )
    if not entries:
        log_check(
            False,
            f"Op-id window suppressed: no Inspector entries in range [{min_id}, {max_id}].",
        )
        return None

    # Render via rich.Table — emitted directly so the script returns None to triage's
    # auto-renderer (its standard list-of-dataclasses pipeline doesn't fit a "context"
    # table whose columns are non-uniform).
    from rich.console import Console
    from rich.table import Table

    table = Table(
        title=f"Op Id context window (running ids ± {window})",
        title_justify="left",
    )
    table.add_column("Op Id", justify="right")
    table.add_column("Status")
    table.add_column("Op Name")
    table.add_column("Devices")

    for raw_id, op_info in entries:
        status = "RUNNING" if raw_id in running_raw_ids else ""
        device_labels = sorted(raw_to_devices.get(raw_id, set()), key=_device_sort_key)
        if full_devices or len(device_labels) <= MAX_DEVICES_DISPLAYED:
            devices_cell = ", ".join(device_labels)
        else:
            devices_cell = ", ".join(device_labels[:MAX_DEVICES_DISPLAYED] + ["..."])
        table.add_row(str(raw_id), status, op_info.name or "N/A", devices_cell)

    Console().print(table)
    return None


if __name__ == "__main__":
    run_script()
