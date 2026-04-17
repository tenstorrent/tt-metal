#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    operation_runtime_map

Description:
    Data provider that builds a mapping from Inspector runtime IDs to
    (operation_name, operation_parameters), and exposes a `lookup()` that
    resolves a dispatcher `host_assigned_id` value (as read from the launch
    message mailbox) to the corresponding operation, accounting for the fact
    that the mailbox value may be an `EncodePerDeviceProgramID`-encoded form
    of the raw runtime_id.

    Encoding rules (mirrored from detail::EncodePerDeviceProgramID in
    tt_metal/impl/profiler/tt_metal_profiler.cpp):
      - Fast dispatch: mailbox host_assigned_id == raw runtime_id.
      - Slow dispatch: mailbox host_assigned_id == EncodePerDeviceProgramID(runtime_id, device_id).

    This is intentionally cached with @triage_singleton so the mapping is computed once per
    triage run (per args/context) and reused by other scripts.

Owner:
    miacim
"""

from __future__ import annotations

from triage import ScriptConfig, triage_singleton, run_script, log_check
from inspector_data import run as get_inspector_data, InspectorData

script_config = ScriptConfig(
    data_provider=True,
    depends=["inspector_data"],
)


# Mirrors detail::EncodePerDeviceProgramID / DecodePerDeviceProgramID.
_DEVICE_ID_NUM_BITS = 10
_DEVICE_OP_ID_NUM_BITS = 31


def _decode_base_program_id(encoded: int) -> int:
    """Extract base_program_id from an EncodePerDeviceProgramID-encoded value."""
    return (encoded & ((1 << _DEVICE_OP_ID_NUM_BITS) - 1)) >> _DEVICE_ID_NUM_BITS


class OperationRuntimeMap:
    """Resolves dispatcher `host_assigned_id` values to Inspector-recorded op info."""

    def __init__(self, runtime_id_map: dict[int, tuple[str, str]]):
        self._map = runtime_id_map

    def __len__(self) -> int:
        return len(self._map)

    def lookup(self, host_assigned_id: int, dispatch_mode: str | None) -> tuple[str, str]:
        """Return (operation_name, operation_parameters), or ("", "") if not found."""
        key = _decode_base_program_id(host_assigned_id) if dispatch_mode == "HOST" else host_assigned_id
        return self._map.get(key, ("", ""))


@triage_singleton
def run(args, context) -> OperationRuntimeMap:
    """Build the runtime_id -> (operation_name, operation_parameters) mapping."""
    inspector_data: InspectorData = get_inspector_data(args, context)
    runtime_id_map: dict[int, tuple[str, str]] = {}

    try:
        runtime_entries_result = inspector_data.getMeshWorkloadRuntimeEntries()
        for entry in runtime_entries_result.runtimeEntries:
            op_name = entry.operationName
            op_params = entry.operationParameters
            runtime_id_map[int(entry.runtimeId)] = (op_name, op_params)

        log_check(True, f"Built runtime_id map with {len(runtime_id_map)} operation(s)")
    except Exception as e:
        log_check(False, f"Failed to build runtime_id to operation map: {e}")
        log_check(False, "Operation names and parameters will not be available")

    return OperationRuntimeMap(runtime_id_map)


if __name__ == "__main__":
    run_script()
