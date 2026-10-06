# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recorded (not gated) trace-replay time of one ttGDN layer at one SP rank, synthetic weights.

``1x1`` runs one Galaxy TP4 rank's heads on one chip at 640 tokens (single-chip direct scan); ``SP1-1x4`` and
``SP1-1x4-T1280`` are the 1x4 TP4 comparison geometries (640 / 1280 tokens) of the current GDN implementation
(tt_metal_tracker-g1b.5.10). Method of the current implementation's timing (qwen36 tests/gdn_baseline
``test_gdn_baseline_perf``), so both are timed alike: one warm forward, one capture, then the median of five
synchronized samples of 100 back-to-back replays (10 replays per sample showed host-jitter spread on ~1 ms layers).
Accuracy is owned by test_gdn_accuracy.py; the LoudBox R12 gate by tests/gdn/perf/test_gdn_layer_perf.py.
"""

from __future__ import annotations

import json
import statistics
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

pytestmark = [run_for_blackhole(), pytest.mark.timeout(900)]

_REPETITIONS = 100
_SAMPLES = 5


@pytest.mark.parametrize(
    "mesh_device,device_params,model,layout",
    [
        pytest.param(
            fixture_mesh_shape(LAYOUTS[layout][0]),
            gdn_device_params(LAYOUTS[layout][0]),
            model,
            layout,
            id=f"{model}-{layout}",
        )
        for model in QWEN_GDN_MODELS
        for layout in ("1x1", "SP1-1x4", "SP1-1x4-T1280")
    ],
    indirect=["mesh_device", "device_params"],
)
def test_gdn_layer_trace_time(mesh_device: ttnn.MeshDevice, device_params: dict, model: str, layout: str) -> None:
    spec = registered_gdn_case(model, layout, "single")
    mesh_device = layout_mesh(mesh_device, spec.mesh_shape)
    case = build_gdn_case(spec)
    layer = make_gdn_device_case(mesh_device, case)
    hidden = to_sp_input(case.chunk_hidden(0), mesh_device, spec.sequence_parallel_axis)
    start = make_actual_start(mesh_device, 0)
    state = layer.allocate_state()
    trace = None
    try:
        warm_output, warm_state = layer.forward(hidden, state, start)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(warm_output)
        deallocate_state(warm_state)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        layer.forward(hidden, state, start)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        samples_ms = []
        for _ in range(_SAMPLES):
            begin = time.perf_counter()
            for _ in range(_REPETITIONS):
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples_ms.append((time.perf_counter() - begin) * 1e3 / _REPETITIONS)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
    config = case.config
    tensor_parallel_size = spec.mesh_shape[spec.tensor_parallel_axis]
    print(
        "GDN_LAYER_TIME="
        + json.dumps(
            {
                "case": spec.name,
                "model": model,
                "layout": layout,
                "tokens": spec.chunk_tokens,
                "key_heads_per_chip": config.num_key_heads // tensor_parallel_size,
                "value_heads_per_chip": config.num_value_heads // tensor_parallel_size,
                "hidden": config.hidden_size,
                "repetitions": _REPETITIONS,
                "median_trace_ms": statistics.median(samples_ms),
                "samples_ms": samples_ms,
            },
            sort_keys=True,
        )
    )
