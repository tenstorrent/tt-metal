# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LoudBox performance gate of one ttGDN layer at the LoudBox layouts, synthetic weights (R12).

Method of the KDA layer perf gate (tests/kda/perf/test_layer_perf.py): production layer and program config, the
first chunk of the registered synthetic ``single`` case (prepared weight cache, no CPU reference), one warm forward,
one trace capture, one warm replay, then five synchronized samples of ten back-to-back non-blocking replays; the
gate reads the median sample (ms per replay). References are the median over five independent sessions (one
process each, interleaved across cases, host load average below 5 on 32 CPUs) on the recorded revision; the gate
bounds the median from above only (+3 %), so a speedup never fails it (recalibrate instead).

LB-A: 2x4 mesh, SP2 x TP4, 1280 tokens per chunk. LB-B: 8x1 mesh, SP8 x TP1, 5120 tokens per chunk, one Galaxy TP4
rank's heads. Opt in with ``KDA_PERF_SKU=bh_loudbox`` (shared with the KDA gates): the references hold for the
LoudBox 8x P150 (11x10 worker grid) only.
"""

from __future__ import annotations

import json
import os
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
from models.demos.deepseek_v3_d_p.tests.gdn.device_utils import gdn_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import deallocate_state, to_sp_input
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(900)]

_REPETITIONS = 10
_TIMING_SAMPLES = 5
_PERF_SKU = "bh_loudbox"
_PERF_MARGIN = 0.03
_LOUDBOX_LAYOUTS = ("LB-A", "LB-B")
# Median trace wall ms per (model, layout): provisional values from the g1b.5.15 G2 arm (ABBA harness), to be replaced
# by the five-session calibration.
_PERF_REFERENCE_MS = {
    ("qwen38_27b", "LB-A"): 1.532,
    ("qwen38_27b", "LB-B"): 1.935,
    ("qwen36_35b", "LB-A"): 1.065,
    ("qwen36_35b", "LB-B"): 1.471,
    ("qwen38_2_4t", "LB-A"): 2.967,
    ("qwen38_2_4t", "LB-B"): 3.645,
    ("qwen38_flash_next", "LB-A"): 1.290,
    ("qwen38_flash_next", "LB-B"): 1.754,
}


def _perf_reference_ms(model: str, layout: str) -> float:
    if os.environ.get("KDA_PERF_SKU") != _PERF_SKU:
        raise ValueError(f"set KDA_PERF_SKU={_PERF_SKU} to opt in to this hardware-specific performance gate")
    return _PERF_REFERENCE_MS[(model, layout)]


def _assert_performance(model: str, layout: str, median_wall_ms: float) -> None:
    reference_ms = _perf_reference_ms(model, layout)
    upper = reference_ms * (1.0 + _PERF_MARGIN)
    assert median_wall_ms <= upper, (
        f"{model} {layout} median trace wall {median_wall_ms:.3f} ms exceeds {upper:.3f} ms "
        f"(reference {reference_ms:.3f} ms + {_PERF_MARGIN:.0%})"
    )


def test_gdn_perf_gate_bounds_regressions_only(monkeypatch, expect_error) -> None:
    monkeypatch.setenv("KDA_PERF_SKU", _PERF_SKU)
    reference_ms = _perf_reference_ms("qwen38_27b", "LB-A")
    _assert_performance("qwen38_27b", "LB-A", reference_ms * 1.029)
    _assert_performance("qwen38_27b", "LB-A", reference_ms * 0.5)
    with expect_error(AssertionError, "exceeds"):
        _assert_performance("qwen38_27b", "LB-A", reference_ms * 1.031)
    monkeypatch.delenv("KDA_PERF_SKU")
    with expect_error(ValueError, "KDA_PERF_SKU"):
        _perf_reference_ms("qwen38_27b", "LB-A")


@pytest.mark.parametrize(
    "mesh_device,device_params,model,layout",
    [
        pytest.param(LAYOUTS[layout][0], gdn_device_params(LAYOUTS[layout][0]), model, layout, id=f"{model}-{layout}")
        for model in QWEN_GDN_MODELS
        for layout in _LOUDBOX_LAYOUTS
    ],
    indirect=["mesh_device", "device_params"],
)
def test_synthetic_gdn_perf(mesh_device: ttnn.MeshDevice, device_params: dict, model: str, layout: str) -> None:
    reference_ms = _perf_reference_ms(model, layout)
    spec = registered_gdn_case(model, layout, "single")
    case = build_gdn_case(spec)
    layer = make_gdn_device_case(mesh_device, case)
    hidden = to_sp_input(case.chunk_hidden(0), mesh_device, spec.sequence_parallel_axis)
    actual_start = make_actual_start(mesh_device, 0)
    state = layer.allocate_state()
    trace = output = next_state = None
    try:
        warm_output, warm_state = layer.forward(hidden, state, actual_start)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(warm_output)
        deallocate_state(warm_state)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output, next_state = layer.forward(hidden, state, actual_start)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples_ms = []
        for _ in range(_TIMING_SAMPLES):
            begin = time.perf_counter()
            for _ in range(_REPETITIONS):
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples_ms.append((time.perf_counter() - begin) * 1e3 / _REPETITIONS)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
        for tensor in (output, hidden, actual_start):
            if tensor is not None:
                ttnn.deallocate(tensor)
        for carry in (next_state, state):
            if carry is not None:
                deallocate_state(carry)
    median_wall_ms = statistics.median(samples_ms)
    grid = mesh_device.compute_with_storage_grid_size()
    tensor_parallel_size = spec.mesh_shape[spec.tensor_parallel_axis]
    print(
        "GDN_SYNTHETIC_PERF="
        + json.dumps(
            {
                "case": spec.name,
                "model": model,
                "layout": layout,
                "fabric_config": ttnn.get_fabric_config().name,
                "compute_grid": [grid.x, grid.y],
                "tokens": spec.chunk_tokens,
                "value_heads_per_chip": case.config.num_value_heads // tensor_parallel_size,
                "repetitions": _REPETITIONS,
                "trace_wall_samples_ms": samples_ms,
                "median_trace_wall_ms": median_wall_ms,
                "reference_trace_wall_ms": reference_ms,
                "perf_margin_pct": _PERF_MARGIN * 100.0,
                "host_load_average_1_5_15": list(os.getloadavg()),
            },
            sort_keys=True,
        )
    )
    _assert_performance(model, layout, median_wall_ms)
