#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    operation_runtime_map

Description:
    Data provider that builds a mapping from Inspector runtime IDs to
    OperationInfo(name, parameters, traced), and exposes a `lookup()` that
    resolves a dispatcher `host_assigned_id` value (as read from the launch
    message mailbox) to the corresponding operation, accounting for the fact
    that the mailbox value may be an `EncodePerDeviceProgramID`-encoded form
    of the raw runtime_id.

    Encoding rules (mirrors detail::EncodePerDeviceProgramID in
    tt_metal/impl/profiler/tt_metal_profiler.cpp):

    Slow dispatch (LaunchProgram path in tt_metal/tt_metal.cpp):
        host_assigned_id is always EncodePerDeviceProgramID(runtime_id, device_id).

    Fast dispatch — non-traced workloads (FDMeshCommandQueue::
    update_launch_messages_for_device_profiler):
        Encoded only when rtoptions().get_profiler_enabled() is true
        (TT_METAL_DEVICE_PROFILER env var). Otherwise raw runtime_id.

    Fast dispatch — traced workloads (FDMeshCommandQueue::record_end):
        Encoded whenever the build defines TRACY_ENABLE (default for
        build_metal.sh). No runtime gate.

    Lookup tries the raw value first and falls back to the decoded form as decoded form is lower in value
    and more likely to hit collision.

    This is intentionally cached with @triage_singleton so the mapping is computed once per
    triage run (per args/context) and reused by other scripts.

Owner:
    miacim
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class OperationInfo:
    name: str
    parameters: str
    traced: bool = False

    @classmethod
    def empty(cls) -> "OperationInfo":
        return cls(name="", parameters="", traced=False)


class OperationRuntimeMap:
    """Resolves dispatcher `host_assigned_id` values to Inspector-recorded op info."""

    def __init__(self, runtime_id_map: dict[int, OperationInfo]):
        self._map = runtime_id_map

    def __len__(self) -> int:
        return len(self._map)

    def lookup(self, host_assigned_id: int, dispatch_mode: str | None) -> OperationInfo:
        if host_assigned_id in self._map:
            return self._map[host_assigned_id]
        decoded = _decode_base_program_id(host_assigned_id)
        return self._map.get(decoded, OperationInfo.empty())


@triage_singleton
def run(args, context) -> OperationRuntimeMap:
    """Build the runtime_id -> OperationInfo mapping."""
    inspector_data: InspectorData = get_inspector_data(args, context)
    runtime_id_map: dict[int, OperationInfo] = {}

    try:
        runtime_entries_result = inspector_data.getMeshWorkloadRuntimeEntries()
        for entry in runtime_entries_result.runtimeEntries:
            traced = bool(getattr(entry, "traced", False))
            runtime_id_map[int(entry.runtimeId)] = OperationInfo(
                name=entry.operationName,
                parameters=entry.operationParameters,
                traced=traced,
            )
        log_check(True, f"Built runtime_id map with {len(runtime_id_map)} operation(s)")
    except Exception as e:
        log_check(False, f"Failed to build runtime_id to operation map: {e}")
        log_check(False, "Operation names and parameters will not be available")

    return OperationRuntimeMap(runtime_id_map)


if __name__ == "__main__":
    run_script()
