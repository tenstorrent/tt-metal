# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Profiled runs of one ttGDN layer for the R14 performance reports (measurement drivers, not gates).

Production layer and program config, first chunk of the registered synthetic ``single`` case (prepared weight cache,
no CPU reference). R14 attributes profiled time against the unprofiled R12 gate (test_gdn_layer_perf.py).

* ``test_gdn_layer_profile`` (run under ``scripts/run_safe_pytest.sh --profile``): one compile forward, two eager
  forwards and three trace replays separated by Tracy signposts; the device profiler is read after every phase so no
  replay overflows the per-core marker buffers. Per-op kernel durations come from the profiler's raw per-core log.
* ``test_gdn_layer_realtime``: real-time profiler records of three eager forwards and three trace replays (per program
  and chip: dispatch-side start / end timestamps and kernel sources); prints ``GDN_LAYER_REALTIME=<json>``. A program
  interval starts when dispatch is ready for the program, so it includes dispatch wait; consecutive intervals of a
  replay tile the layer.
"""

from __future__ import annotations

import json
import os
import time

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import QWEN_GDN_MODELS
from models.demos.deepseek_v3_d_p.tests.gdn.cases import (
    LAYOUTS,
    build_gdn_case,
    make_gdn_device_case,
    registered_gdn_case,
)
from models.demos.deepseek_v3_d_p.tests.gdn.device_utils import fixture_mesh_shape, gdn_device_params, layout_mesh
from models.demos.deepseek_v3_d_p.tests.kda.utils import deallocate_state, to_sp_input
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(900)]

_EAGER = 2
_REPLAYS = 3
_RECORD_SETTLE_SECONDS = 0.2
_RECORD_TIMEOUT_SECONDS = 10.0
_PARAMS = [
    pytest.param(
        fixture_mesh_shape(LAYOUTS[layout][0]),
        gdn_device_params(LAYOUTS[layout][0]),
        model,
        layout,
        id=f"{model}-{layout}",
    )
    for model in QWEN_GDN_MODELS
    for layout in LAYOUTS
    if layout != "1x1"
]


class _Layer:
    """The case's production layer, its first-chunk input and a zero carry."""

    def __init__(self, mesh_device: ttnn.MeshDevice, model: str, layout: str):
        self.spec = registered_gdn_case(model, layout, "single")
        self.mesh = layout_mesh(mesh_device, self.spec.mesh_shape)
        self.case = build_gdn_case(self.spec)
        self.layer = make_gdn_device_case(self.mesh, self.case)
        self.hidden = to_sp_input(self.case.chunk_hidden(0), self.mesh, self.spec.sequence_parallel_axis)
        self.start = make_actual_start(self.mesh, 0)
        self.state = self.layer.allocate_state()

    def forward(self) -> tuple[ttnn.Tensor, object]:
        return self.layer.forward(self.hidden, self.state, self.start)

    def eager(self) -> None:
        output, next_state = self.forward()
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(output)
        deallocate_state(next_state)

    def environment(self) -> dict:
        from models.tt_transformers.tt.ccl import get_num_links

        grid = self.mesh.compute_with_storage_grid_size()
        tensor_parallel_size = self.spec.mesh_shape[self.spec.tensor_parallel_axis]
        return {
            "case": self.spec.name,
            "mesh_shape": list(self.spec.mesh_shape),
            "fabric_config": ttnn.get_fabric_config().name,
            "compute_grid": [grid.x, grid.y],
            "tokens": self.spec.chunk_tokens,
            "key_heads_per_chip": self.case.config.num_key_heads // tensor_parallel_size,
            "value_heads_per_chip": self.case.config.num_value_heads // tensor_parallel_size,
            "fabric_links_per_axis": [
                get_num_links(self.mesh, axis) if self.spec.mesh_shape[axis] > 1 else 0 for axis in (0, 1)
            ],
            "host_load_average_1_5_15": list(os.getloadavg()),
        }

    def close(self) -> None:
        ttnn.deallocate(self.hidden)
        ttnn.deallocate(self.start)
        deallocate_state(self.state)


@pytest.mark.parametrize("mesh_device,device_params,model,layout", _PARAMS, indirect=["mesh_device", "device_params"])
def test_gdn_layer_profile(mesh_device: ttnn.MeshDevice, device_params: dict, model: str, layout: str) -> None:
    from tracy import signpost

    run = _Layer(mesh_device, model, layout)
    signpost("gdn_compile")
    run.eager()
    ttnn.ReadDeviceProfiler(run.mesh)
    for index in range(_EAGER):
        signpost(f"gdn_eager{index}")
        run.eager()
        ttnn.ReadDeviceProfiler(run.mesh)
    trace = ttnn.begin_trace_capture(run.mesh, cq_id=0)
    output, next_state = run.forward()
    ttnn.end_trace_capture(run.mesh, trace, cq_id=0)
    try:
        for index in range(_REPLAYS):
            signpost(f"gdn_trace_replay{index}")
            ttnn.execute_trace(run.mesh, trace, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(run.mesh)
        signpost("gdn_end")
    finally:
        ttnn.release_trace(run.mesh, trace)
        ttnn.deallocate(output)
        deallocate_state(next_state)
        run.close()
    print("GDN_LAYER_PROFILE=" + json.dumps({"model": model, "layout": layout, **run.environment()}, sort_keys=True))


def _realtime_records(mesh: ttnn.MeshDevice, run_fn) -> list[dict]:
    """Every real-time profiler record of ``run_fn``'s programs on every chip, in arrival order."""
    records, dropped = [], [0]

    def collect(batch) -> None:
        dropped[0] += int(batch.dropped)
        for record in batch.records:
            records.append(
                {
                    "runtime_id": int(record.runtime_id),
                    "chip_id": int(record.chip_id),
                    "start": int(record.start_timestamp),
                    "end": int(record.end_timestamp),
                    "frequency_ghz": float(record.frequency),
                    "kernels": sorted({str(source).rsplit("/", 1)[-1] for source in record.kernel_sources}),
                }
            )

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
    try:
        run_fn()
        ttnn.synchronize_device(mesh)
        deadline = time.monotonic() + _RECORD_TIMEOUT_SECONDS
        count, changed = 0, time.monotonic()
        while time.monotonic() < deadline:
            if len(records) != count:
                count, changed = len(records), time.monotonic()
            elif count and time.monotonic() - changed >= _RECORD_SETTLE_SECONDS:
                break
            time.sleep(0.01)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    assert not dropped[0], f"real-time profiler dropped {dropped[0]} records"
    assert records, "real-time profiler delivered no records"
    return records


@pytest.mark.parametrize("mesh_device,device_params,model,layout", _PARAMS, indirect=["mesh_device", "device_params"])
def test_gdn_layer_realtime(mesh_device: ttnn.MeshDevice, device_params: dict, model: str, layout: str) -> None:
    from tests.ttnn.profiling.realtime_profiler_utils import require_realtime_profiler

    require_realtime_profiler("GDN layer R14 realtime records")
    run = _Layer(mesh_device, model, layout)
    run.eager()
    run.eager()
    eager = [_realtime_records(run.mesh, run.eager) for _ in range(_EAGER + 1)]
    trace = ttnn.begin_trace_capture(run.mesh, cq_id=0)
    output, next_state = run.forward()
    ttnn.end_trace_capture(run.mesh, trace, cq_id=0)
    try:
        ttnn.execute_trace(run.mesh, trace, cq_id=0, blocking=True)
        traced = [
            _realtime_records(run.mesh, lambda: ttnn.execute_trace(run.mesh, trace, cq_id=0, blocking=False))
            for _ in range(_REPLAYS)
        ]
    finally:
        ttnn.release_trace(run.mesh, trace)
        ttnn.deallocate(output)
        deallocate_state(next_state)
        run.close()
    print(
        "GDN_LAYER_REALTIME="
        + json.dumps(
            {"model": model, "layout": layout, **run.environment(), "eager": eager, "traced": traced},
            sort_keys=True,
        )
    )
