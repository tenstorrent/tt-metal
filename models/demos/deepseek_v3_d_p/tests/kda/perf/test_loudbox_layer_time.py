# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer trace-replay time and per-program device time of the production KDA layer at the LoudBox layouts.

Measurement harness for paired revision A/B comparisons, not a gate (it asserts only finite output). One item
measures one model x layout of the production layer (production program config, synthetic weights, first chunk,
no CPU reference). Two revisions differ in code, so they cannot share a process: compare them across alternating
sessions and pair adjacent sessions (tt_metal_tracker-g1b.7, tt_metal_tracker-g1b.4.15).

- ``test_loudbox_layer_time``: 2 warm forwards, one trace capture, then ``_SAMPLES`` samples of ``_REPETITIONS``
  non-blocking replays each; prints ``KDA_LAYER_TIME=<json>`` with every sample (ms per replay). Set
  ``KDA_LAYER_OUTPUT_DUMP=<path>`` to save the first replay's output for a cross-revision comparison.
- ``test_loudbox_layer_op_breakdown``: realtime-profiler duration (max over chips) of every program of 3 eager
  forwards and 3 trace replays, in dispatch order; prints ``KDA_LAYER_OPS=<json>``.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.cases import build_kda_case, loudbox_kda_case, make_kda_device_case
from models.demos.deepseek_v3_d_p.tests.kda.utils import deallocate_state
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(900)]

_LAYOUTS = {"LB-A": (2, 4), "LB-B": (8, 1)}
_MODELS = ("kimi_k3", "glm_5_3_flash")
_REPETITIONS = 10
_SAMPLES = 16
_PARAMS = [
    pytest.param(_LAYOUTS[layout], fabric_1d_device_params(), model, layout, id=f"{model}-{layout}")
    for model in _MODELS
    for layout in _LAYOUTS
]


def _host_output(output: ttnn.Tensor) -> torch.Tensor:
    return torch.cat([ttnn.to_torch(shard).float().flatten() for shard in ttnn.get_device_tensors(output)])


def _production_layer(mesh_device: ttnn.MeshDevice, model: str, layout: str):
    case = build_kda_case(loudbox_kda_case("synthetic", layout, "single", model=model))
    layer, hidden = make_kda_device_case(mesh_device, case)
    return layer, hidden, layer.allocate_state(batch_size=1), make_actual_start(mesh_device, 0)


def _warm(mesh_device: ttnn.MeshDevice, layer, hidden, state, actual_start) -> None:
    for _ in range(2):
        output, next_state = layer.forward(hidden, state, actual_start)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
        deallocate_state(next_state)


@pytest.mark.parametrize("mesh_device,device_params,model,layout", _PARAMS, indirect=["mesh_device", "device_params"])
def test_loudbox_layer_time(mesh_device: ttnn.MeshDevice, device_params: dict, model: str, layout: str) -> None:
    layer, hidden, state, actual_start = _production_layer(mesh_device, model, layout)
    _warm(mesh_device, layer, hidden, state, actual_start)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output, next_state = layer.forward(hidden, state, actual_start)
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        first_output = _host_output(output)
        dump = os.environ.get("KDA_LAYER_OUTPUT_DUMP")
        if dump:
            torch.save(first_output.bfloat16(), dump)
        samples = []
        for _ in range(_SAMPLES):
            start = time.perf_counter()
            for _ in range(_REPETITIONS):
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - start) * 1e3 / _REPETITIONS)
    finally:
        ttnn.release_trace(mesh_device, trace)
        ttnn.deallocate(output)
        deallocate_state(next_state)
        deallocate_state(state)
    result = {
        "model": model,
        "layout": layout,
        "repetitions": _REPETITIONS,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "output_finite": bool(torch.isfinite(first_output).all()),
    }
    print("KDA_LAYER_TIME=" + json.dumps(result, sort_keys=True))
    assert result["output_finite"]


@pytest.mark.parametrize("mesh_device,device_params,model,layout", _PARAMS, indirect=["mesh_device", "device_params"])
def test_loudbox_layer_op_breakdown(mesh_device: ttnn.MeshDevice, device_params: dict, model: str, layout: str) -> None:
    from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

    require_realtime_profiler("KDA layer op breakdown")
    layer, hidden, state, actual_start = _production_layer(mesh_device, model, layout)
    _warm(mesh_device, layer, hidden, state, actual_start)

    def summarize(per_program: dict) -> list:
        return [
            {
                "seq": seq,
                "runtime_id": runtime_id,
                "duration_ns": entry["duration_ns"],
                "kernels": sorted({source.rsplit("/", 1)[-1] for source in entry["kernel_sources"]}),
            }
            for seq, (runtime_id, entry) in enumerate(per_program.items())
        ]

    eager = []
    for _ in range(3):
        outputs, per_program = profile_realtime_program_merged(
            mesh_device, lambda: layer.forward(hidden, state, actual_start), record_timeout_seconds=5.0
        )
        output, next_state = outputs
        ttnn.deallocate(output)
        deallocate_state(next_state)
        eager.append(summarize(per_program))
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output, next_state = layer.forward(hidden, state, actual_start)
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    traced = []
    try:
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        for _ in range(3):
            try:
                _, per_program = profile_realtime_program_merged(
                    mesh_device,
                    lambda: ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False),
                    record_timeout_seconds=5.0,
                )
                traced.append(summarize(per_program))
            except (RuntimeError, AssertionError) as error:  # a traced replay may not emit realtime records
                traced.append({"error": str(error)})
    finally:
        ttnn.release_trace(mesh_device, trace)
        ttnn.deallocate(output)
        deallocate_state(next_state)
        deallocate_state(state)
    print("KDA_LAYER_OPS=" + json.dumps({"model": model, "layout": layout, "eager": eager, "traced": traced}))
