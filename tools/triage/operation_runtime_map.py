#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    operation_runtime_map

Description:
    Data provider that builds a mapping from Inspector runtime IDs to `OperationInfo`
    (name, parameters, trace_id) and exposes a `lookup()` that resolves a dispatcher
    `host_assigned_id` value (as read from the launch message mailbox) to the corresponding
    operation, or None if no entry is found.

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

# Matches the sentinel in rpc.capnp (kNoTraceId). Capnp has no null for primitives.
_NO_TRACE_ID = 0xFFFFFFFF


def _decode_base_program_id(encoded: int) -> int:
    """Extract base_program_id from an EncodePerDeviceProgramID-encoded value."""
    return (encoded & ((1 << _DEVICE_OP_ID_NUM_BITS) - 1)) >> _DEVICE_ID_NUM_BITS


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

    def lookup(self, host_assigned_id: int, dispatch_mode: str | None) -> OperationInfo | None:
        """Resolve host_assigned_id to its Inspector entry, or None if not found.

        - Slow dispatch (HOST): decode first; the mailbox always holds the encoded form.
        - Fast dispatch (DEV) or unknown: try raw first, then decoded as a fallback (covers
          the FD-with-profiler case where the encoded form ends up in the mailbox too).
        """
        if dispatch_mode == "HOST":
            return self._map.get(_decode_base_program_id(host_assigned_id))

        entry = self._map.get(host_assigned_id)
        if entry is not None:
            return entry

        return self._map.get(_decode_base_program_id(host_assigned_id))


@triage_singleton
def run(args, context) -> OperationRuntimeMap:
    """Build the runtime_id -> OperationInfo mapping."""
    inspector_data: InspectorData = get_inspector_data(args, context)
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


if __name__ == "__main__":
    run_script()
