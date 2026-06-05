#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_op_window [--op-window=<n>]

Options:
    --op-window=<n>   Render ops with id in [min_running_op_id - n, max_running_op_id + n].
                      Use this to see what ran just before / what should have run next around
                      the currently-running set. [default: 10]

Description:
    Companion to `dump_running_operations`. Reads the same aggregation, then emits a
    context-window table sourced from the Inspector runtime map. One row per Inspector
    entry in the window:
      - Op Id (raw runtime id)
      - Status (`RUNNING` if currently observed in dispatcher mailboxes, blank otherwise)
      - Op Name (Inspector)
      - Devices (sorted device labels currently running this op id; blank for neighbors)

Owner:
    onenezicTT
"""

from __future__ import annotations

from dataclasses import dataclass

from operation_provider import (
    run as get_operation_provider,
    OperationRuntimeMap,
    RunningOperationAggregation,
)
from triage import (
    ScriptConfig,
    ScriptPriority,
    collection_serializer,
    log_warning,
    run_script,
    triage_field,
)
from ttexalens.context import Context


script_config = ScriptConfig(
    depends=["operation_provider"],
    priority=ScriptPriority.HIGH,
)


@dataclass
class OpWindowRow:
    op_id: int = triage_field("Op Id")
    status: str = triage_field("Status")
    op_name: str = triage_field("Op Name")
    devices: list[str] = triage_field("Devices", collection_serializer(", "))


def _build_raw_id_to_devices(
    aggregations: dict[int, RunningOperationAggregation],
    runtime_id_to_operation: OperationRuntimeMap,
) -> dict[int, set[str]]:
    raw_to_devices: dict[int, set[str]] = {}
    for host_assigned_id, agg in aggregations.items():
        if runtime_id_to_operation.lookup(host_assigned_id) is None:
            continue
        raw_to_devices.setdefault(host_assigned_id, set()).update(agg.device_labels)
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

    if window <= 0:
        log_warning(f"Op-id window suppressed: --op-window={window} (must be > 0).")
        return None

    bundle = get_operation_provider(args, context)
    aggregations = bundle.aggregations
    runtime_id_to_operation = bundle.runtime_id_to_operation

    if not aggregations:
        return None

    raw_to_devices = _build_raw_id_to_devices(aggregations, runtime_id_to_operation)
    running_raw_ids = set(raw_to_devices.keys())
    if not running_raw_ids:
        log_warning(
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
        log_warning(f"Op-id window suppressed: no Inspector entries in range [{min_id}, {max_id}].")
        return None

    rows: list[OpWindowRow] = []
    for raw_id, op_info in entries:
        device_labels = sorted(raw_to_devices.get(raw_id, set()), key=_device_sort_key)
        rows.append(
            OpWindowRow(
                op_id=raw_id,
                status="RUNNING" if raw_id in running_raw_ids else "",
                op_name=op_info.name or "N/A",
                devices=device_labels,
            )
        )
    return rows


if __name__ == "__main__":
    run_script()
