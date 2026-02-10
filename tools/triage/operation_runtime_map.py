#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    operation_runtime_map

Description:
    Data provider that builds a mapping from dispatcher runtime IDs (host_assigned_id) to
    (operation_name, operation_parameters) using Inspector mesh workloads.

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


@triage_singleton
def run(args, context) -> dict[int, tuple[str, str]]:
    """Return runtime_id -> (operation_name, operation_parameters).

    Assumes host_assigned_id (dispatcher) == runtime_id (inspector).
    Always returns a dict (possibly empty) so downstream scripts can degrade gracefully.
    """
    inspector_data: InspectorData = get_inspector_data(args, context)
    runtime_id_map: dict[int, tuple[str, str]] = {}

    try:
        mesh_workloads_result = inspector_data.getMeshWorkloads()
        mesh_workloads = {w.meshWorkloadId: w for w in mesh_workloads_result.meshWorkloads}

        runtime_ids_result = inspector_data.getMeshWorkloadsRuntimeIds()
        for entry in runtime_ids_result.runtimeIds:
            workload = mesh_workloads.get(entry.workloadId)
            if workload is None:
                continue
            name = getattr(entry, "name", "") or (workload.name if workload else "")
            params = getattr(entry, "parameters", "") or (workload.parameters if workload else "")

            runtime_id_map[int(entry.runtimeId)] = (name, params)
        log_check(True, f"Built runtime_id map with {len(runtime_id_map)} operation(s)")
    except Exception as e:
        log_check(False, f"Failed to build runtime_id to operation map: {e}")
        log_check(False, "Operation names and parameters will not be available")

    return runtime_id_map


if __name__ == "__main__":
    run_script()
