# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared allocation and device-program profiling for KDA experiments."""

import json
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ttnn
from models.demos.deepseek_v3_d_p.tests.kda.utils import _deallocate_state, make_actual_start
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program


def capture_resources(run: Callable, mesh_device: ttnn.MeshDevice, label: str):
    destination = os.environ.get("KDA_EVIDENCE_DIR")
    if destination is None:
        return None
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        result = run()
        ttnn.synchronize_device(mesh_device)
    finally:
        graph = ttnn.graph.end_graph_capture()
    path = Path(destination) / f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2) + "\n")
    live = {}
    peaks = defaultdict(int)
    operations = Counter()
    for node in graph:
        params = node.get("params", {})
        kind = node["node_type"]
        if kind == "function_start":
            operations[params.get("name", "unknown")] += 1
        if kind not in ("buffer_allocate", "buffer_deallocate"):
            continue
        identity = (params["device_id"], params["type"], params["address"])
        if kind == "buffer_allocate":
            banks = ttnn.get_memory_view(
                mesh_device, ttnn.BufferType.DRAM if params["type"] == "DRAM" else ttnn.BufferType.L1
            ).num_banks
            live[identity] = (int(params["size"]), int(params["max_size_per_bank"]) * banks)
        else:
            live.pop(identity, None)
        for device, memory, _ in live:
            for index, metric in enumerate(("requested", "bank_aligned")):
                key = f"device{device}_{memory}_{metric}"
                total = sum(size[index] for (d, m, _), size in live.items() if d == device and m == memory)
                peaks[key] = max(peaks[key], total)
    print(
        "KDA_RESOURCE_EXPERIMENT="
        + json.dumps(
            dict(label=label, peak_new_allocation_bytes=dict(peaks), operation_calls=dict(operations), graph=str(path)),
            sort_keys=True,
        )
    )
    return result


def profile_resources(run: Callable, mesh_device: ttnn.MeshDevice, label: str):
    destination = os.environ["KDA_EVIDENCE_DIR"]
    from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler

    require_realtime_profiler(label)
    result, records = profile_realtime_program(mesh_device, run, collect_all=True, record_timeout_seconds=30)
    (Path(destination) / f"{label}.programs.json").write_text(json.dumps(records, indent=2) + "\n")
    programs = {}
    for record in records:
        key = record["runtime_id"]
        if not key:
            continue
        item = programs.setdefault(key, dict(max_ns=0, chips=set(), kernels=record["kernel_sources"]))
        item["max_ns"] = max(item["max_ns"], record["duration_ns"])
        item["chips"].add(record["chip_id"])
    assert programs
    assert all(len(item["chips"]) == mesh_device.get_num_devices() for item in programs.values())
    print(
        "KDA_PROGRAM_EXPERIMENT="
        + json.dumps(
            dict(
                label=label,
                programs=[
                    dict(max_ns=item["max_ns"], chips=sorted(item["chips"]), kernels=item["kernels"])
                    for item in programs.values()
                ],
            ),
            sort_keys=True,
        )
    )
    return result


def _device_program_label(kernel_sources: tuple[str, ...]) -> str:
    names = set()
    for source in kernel_sources:
        parts = source.replace("\\", "/").split("/")
        if "operations" not in parts:
            continue
        index = parts.index("operations") + 1
        experimental = index < len(parts) and parts[index] == "experimental"
        if experimental:
            index += 1
        if index < len(parts):
            name = f"{'experimental.' if experimental else ''}{parts[index]}"
            if name.endswith(("kda", "ccl")) and index + 1 < len(parts):
                name = f"{name}.{parts[index + 1]}"
            names.add(name)
    if names:
        return "+".join(sorted(names))
    basenames = {Path(source).stem for source in kernel_sources}
    return "+".join(sorted(basenames)) if basenames else "unknown"


def _log_device_program_times(
    mesh_device: ttnn.MeshDevice,
    layer: ttKDA,
    hidden: ttnn.Tensor,
    layout: str,
    *,
    actual_start: int = 0,
) -> list[dict[str, Any]]:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        raise RuntimeError(f"real-time profiler is inactive for the {layout} KDA e2e device-time breakdown")
    state = layer.allocate_state(batch_size=1)
    output = None
    next_state = None
    profiled_results: list[tuple[ttnn.Tensor, KdaState]] = []

    actual_start_tt = make_actual_start(mesh_device, actual_start)

    def run_profiled_forward() -> tuple[ttnn.Tensor, KdaState]:
        result = layer.forward(hidden, state, actual_start_tt)
        profiled_results.append(result)
        return result

    try:
        (output, next_state), records = profile_realtime_program(
            mesh_device,
            run_profiled_forward,
            collect_all=True,
            record_timeout_seconds=30.0,
        )
        per_program: dict[int, dict[str, Any]] = {}
        for record in records:
            runtime_id = record["runtime_id"]
            if not runtime_id:
                continue
            entry = per_program.setdefault(
                runtime_id,
                {
                    "duration_ns": 0.0,
                    "kernel_sources": record["kernel_sources"],
                    "chip_ids": set(),
                    "record_count": 0,
                },
            )
            entry["duration_ns"] = max(entry["duration_ns"], record["duration_ns"])
            entry["chip_ids"].add(record["chip_id"])
            entry["record_count"] += 1
        if not per_program:
            raise RuntimeError("real-time profiler returned no KDA program records")
        expected_chip_count = mesh_device.get_num_devices()
        programs: list[dict[str, Any]] = [
            {
                "sequence": sequence,
                "name": _device_program_label(info["kernel_sources"]),
                "device_time_ns": round(float(info["duration_ns"]), 3),
                "chip_count": len(info["chip_ids"]),
                "record_count": int(info["record_count"]),
                "complete": len(info["chip_ids"]) == expected_chip_count,
            }
            for sequence, info in enumerate(per_program.values())
        ]
        # Summary and scan intentionally share one device-operation factory and
        # therefore the same kernel source paths.  In a KDA layer they are the
        # first and second occurrence, respectively; name the first explicitly
        # so topology and timings remain attributable to the two distinct calls.
        recurrent_programs = [
            program for program in programs if program["name"] == "experimental.kda.recurrent_chunk_scan"
        ]
        if len(recurrent_programs) == 2:
            recurrent_programs[0]["name"] = "experimental.kda.summarize_chunk_recurrence"
        incomplete_program_sequences = [program["sequence"] for program in programs if not program["complete"]]
        durations_by_name: dict[str, list[float]] = {}
        for program in programs:
            durations_by_name.setdefault(program["name"], []).append(program["device_time_ns"])
        operation_summary = [
            {
                "name": name,
                "program_count": len(durations),
                "median_device_time_ns": round(statistics.median(durations), 3),
                "max_device_time_ns": round(max(durations), 3),
            }
            for name, durations in durations_by_name.items()
        ]
        print(
            "KDA_LAYER_DEVICE_TIMES="
            + json.dumps(
                {
                    "layout": layout,
                    "actual_start": actual_start,
                    "measurement": "one warm eager forward outside gated trace samples",
                    "duration_semantics": (
                        "per-program max across reported chip records; programs may overlap and durations must not be summed"
                    ),
                    "chip_completeness": {
                        "expected_chip_count": expected_chip_count,
                        "incomplete_program_sequences": incomplete_program_sequences,
                    },
                    "operation_summary": operation_summary,
                    "programs": programs,
                },
                sort_keys=True,
            )
        )
        return programs
    finally:
        if profiled_results and output is None:
            output, next_state = profiled_results[-1]
        if output is not None:
            ttnn.deallocate(output)
        if next_state is not None:
            _deallocate_state(next_state)
        _deallocate_state(state)
