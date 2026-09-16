# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optional raw allocation evidence outside the checkout for carry experiments."""
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import ttnn


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
    if os.environ.get("KDA_PROFILE_CARRY") == "1":
        from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler

        require_realtime_profiler(label)
        for tensor in result if isinstance(result, tuple) else (result,):
            ttnn.deallocate(tensor)
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
