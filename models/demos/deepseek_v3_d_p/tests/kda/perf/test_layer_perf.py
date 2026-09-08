# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local real-weight and CI-synthetic performance acceptance for Kimi-K3 KDA."""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import load_or_compute_cpu_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    check_kimi_k3_accuracy,
    deallocate_state,
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
    make_synthetic_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(900),
]

_SEQUENCE = 5120
_REPETITIONS = 10
_TIMING_SAMPLES = 5
_PCC_THRESHOLD = 0.9995
_PERF_SKU = "bh_loudbox"
_PERF_MARGIN = 0.03
# LoudBox calibration at 350413d7a98e (2026-08-31): median across five independent
# sessions, each using the median of five warm synchronized 10-replay samples.
_PERF_REFERENCE_MS = {
    "SP1xTP8": 9.597,
    "SP2xTP4": 9.539,
    "SP4xTP2": 9.991,
}
# Blackhole Galaxy SP8xTP4 calibration at c4f8ddd0e377 (2026-09-02): median
# of five warm synchronized 10-replay samples on the high-power CI lane.
_GALAXY_PERF_REFERENCE_MS = 3.963


@pytest.fixture(scope="session")
def kimi_k3_production_reference(
    kimi_k3_checkpoint_dir: Path,
) -> Callable[[], tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float]]:
    """Return a lazy loader for the session-cached production-length CPU oracle."""
    cached_reference: tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float] | None = None

    def load() -> tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float]:
        nonlocal cached_reference
        if cached_reference is None:
            case = make_kimi_k3_test_case(kimi_k3_checkpoint_dir, sequence=_SEQUENCE)
            golden_output, golden_state, elapsed = load_or_compute_cpu_reference(case)
            cached_reference = case, golden_output, golden_state, elapsed
        return cached_reference

    return load


def _perf_reference_ms(layout: str) -> float:
    if os.environ.get("KDA_PERF_SKU") != _PERF_SKU:
        raise ValueError(f"set KDA_PERF_SKU={_PERF_SKU} to opt in to this hardware-specific performance gate")
    return _PERF_REFERENCE_MS[layout]


def _allocate_state(layer: ttKDA) -> KdaState:
    return layer.allocate_state(batch_size=1)


def _synthetic_perf_reference_ms(layout: str) -> float:
    if layout == "SP8xTP4":
        return _GALAXY_PERF_REFERENCE_MS
    return _perf_reference_ms(layout)


def _assert_synthetic_performance(layout: str, median_wall_ms: float) -> None:
    reference_ms = _synthetic_perf_reference_ms(layout)
    lower = reference_ms * (1.0 - _PERF_MARGIN)
    upper = reference_ms * (1.0 + _PERF_MARGIN)
    assert lower <= median_wall_ms <= upper, (
        f"{layout} median trace wall {median_wall_ms:.3f} ms is outside performance range "
        f"[{lower:.3f}, {upper:.3f}] ms "
        f"(reference {reference_ms:.3f} ms ± {_PERF_MARGIN:.0%})"
    )


@pytest.mark.parametrize("layout", ["SP2xTP4", "SP8xTP4"])
def test_synthetic_performance_uses_two_sided_margin(layout, monkeypatch, expect_error) -> None:
    monkeypatch.setenv("KDA_PERF_SKU", _PERF_SKU)
    reference_ms = _synthetic_perf_reference_ms(layout)
    _assert_synthetic_performance(layout, reference_ms)
    with expect_error(AssertionError, "outside performance range"):
        _assert_synthetic_performance(layout, reference_ms * 0.96)
    with expect_error(AssertionError, "outside performance range"):
        _assert_synthetic_performance(layout, reference_ms * 1.04)


def _trace_wall_samples_ms(
    mesh_device: ttnn.MeshDevice,
    layer: ttKDA,
    hidden: ttnn.Tensor,
    repetitions: int,
    validate_first_replay: Callable[[KdaState, ttnn.Tensor], dict[str, float]] | None = None,
    actual_start: int = 0,
) -> tuple[list[float], dict[str, float] | None]:
    state = None
    warm_output = None
    warm_state = None
    trace_id = None
    output = None
    next_state = None
    actual_start_tt = make_actual_start(mesh_device, actual_start)
    try:
        state = _allocate_state(layer)
        warm_output, warm_state = layer.forward(hidden, state, actual_start_tt)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(warm_output)
        warm_output = None
        deallocate_state(warm_state)
        warm_state = None

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output, next_state = layer.forward(hidden, state, actual_start_tt)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        validation = validate_first_replay(next_state, output) if validate_first_replay is not None else None

        samples_ms = []
        for _ in range(_TIMING_SAMPLES):
            start = time.perf_counter()
            for _ in range(repetitions):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples_ms.append((time.perf_counter() - start) * 1e3 / repetitions)
        return samples_ms, validation
    finally:
        try:
            if trace_id is not None:
                ttnn.release_trace(mesh_device, trace_id)
        finally:
            if warm_output is not None:
                ttnn.deallocate(warm_output)
            if warm_state is not None:
                deallocate_state(warm_state)
            if output is not None:
                ttnn.deallocate(output)
            if state is not None:
                deallocate_state(state)
            if next_state is not None:
                deallocate_state(next_state)


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0)],
    indirect=["mesh_device"],
    ids=["SP1xTP8", "SP2xTP4", "SP4xTP2"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        # FABRIC_2D follow-up: SP1xTP8 hangs with a device timeout and failed
        # Ethernet-core recovery. SP2xTP4/SP4xTP2 were correct but 0.05%/0.10%
        # slower than FABRIC_1D; do not enable 2D until the SP1 failure is fixed.
        pytest.param(
            fabric_1d_device_params(model_config=KimiK3Config),
            id="fabric_1d",
        ),
    ],
    indirect=True,
)
def test_kimi_k3_layer_1_perf(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_production_reference: Callable[[], tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float]],
) -> None:
    """Compare production geometry with an independent CPU oracle before timing it."""
    sequence = _SEQUENCE
    sequence_parallel_axis = 1 - tensor_parallel_axis
    mesh_shape = tuple(mesh_device.shape)
    layout = f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}"
    repetitions = _REPETITIONS
    reference_ms = _perf_reference_ms(layout)
    case, golden_output, golden_state, cpu_reference_seconds = kimi_k3_production_reference()
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
    )

    initial_state = _allocate_state(layer)
    start = time.perf_counter()
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, state = layer.forward(hidden_tt, initial_state, make_actual_start(layer.device))
    ttnn.synchronize_device(mesh_device)
    device_forward_ms = (time.perf_counter() - start) * 1e3
    try:
        pcc = check_kimi_k3_accuracy(
            f"Kimi-K3 layer 1 T={sequence} {layout}",
            case,
            golden_output,
            golden_state,
            state,
            output,
            mesh_device,
            tensor_parallel_axis,
            pcc_threshold=_PCC_THRESHOLD,
        )
    finally:
        ttnn.deallocate(output)
    deallocate_state(initial_state)
    deallocate_state(state)

    validate_trace_replay = partial(
        check_kimi_k3_accuracy,
        f"Kimi-K3 layer 1 T={case.hidden.shape[1]} {layout} trace replay",
        case,
        golden_output,
        golden_state,
        mesh_device=mesh_device,
        tensor_parallel_axis=tensor_parallel_axis,
        pcc_threshold=_PCC_THRESHOLD,
    )
    samples_ms, trace_pcc = _trace_wall_samples_ms(
        mesh_device,
        layer,
        hidden_tt,
        repetitions,
        validate_first_replay=validate_trace_replay,
    )
    assert trace_pcc is not None
    first_wall_ms = samples_ms[0]
    median_wall_ms = statistics.median(samples_ms)
    tail_wall_ms = max(samples_ms)
    min_wall_ms = reference_ms * (1.0 - _PERF_MARGIN)
    max_wall_ms = reference_ms * (1.0 + _PERF_MARGIN)
    result = {
        "fabric_config": ttnn.get_fabric_config().name,
        "layout": layout,
        "sequence": sequence,
        "repetitions": repetitions,
        "pcc": pcc,
        "trace_pcc": trace_pcc,
        "pcc_reference": "independent pure-Torch FP32 CPU reference",
        "trace_wall_ms": median_wall_ms,
        "first_trace_wall_ms": first_wall_ms,
        "trace_wall_samples_ms": samples_ms,
        "median_trace_wall_ms": median_wall_ms,
        "tail_trace_wall_ms": tail_wall_ms,
        "timing_sample_count": _TIMING_SAMPLES,
        "reference_trace_wall_ms": reference_ms,
        "perf_margin_pct": _PERF_MARGIN * 100.0,
        "min_trace_wall_ms": min_wall_ms,
        "max_trace_wall_ms": max_wall_ms,
        "cpu_reference_seconds": cpu_reference_seconds,
        "device_forward_ms": device_forward_ms,
    }
    print("KDA_LAYER_PERF=" + json.dumps(result, sort_keys=True))

    assert min_wall_ms <= median_wall_ms <= max_wall_ms, (
        f"{layout} median trace wall {median_wall_ms:.3f} ms is outside LoudBox range "
        f"[{min_wall_ms:.3f}, {max_wall_ms:.3f}] ms (reference {reference_ms:.3f} ms ± {_PERF_MARGIN:.0%})"
    )


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param(
            (2, 4),
            1,
            fabric_1d_device_params(
                model_config=KimiK3Config,
            ),
            id="SP2xTP4-fabric-1d",
        ),
        pytest.param(
            (8, 4),
            1,
            torus_xy_device_params(
                model_config=KimiK3Config,
            ),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_synthetic_kimi_k3_perf(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    device_params: dict,
) -> None:
    """Gate checkpoint-free production K3 latency on LoudBox and Galaxy."""
    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    layout = f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}"
    reference_ms = _synthetic_perf_reference_ms(layout)
    case = make_synthetic_kimi_k3_test_case(sequence=_SEQUENCE)
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )
    samples_ms, _ = _trace_wall_samples_ms(mesh_device, layer, hidden_tt, _REPETITIONS)
    median_wall_ms = statistics.median(samples_ms)
    result = {
        "fabric_config": ttnn.get_fabric_config().name,
        "layout": layout,
        "sequence": _SEQUENCE,
        "weights": "deterministic synthetic",
        "repetitions": _REPETITIONS,
        "trace_wall_samples_ms": samples_ms,
        "median_trace_wall_ms": median_wall_ms,
        "timing_sample_count": _TIMING_SAMPLES,
        "reference_trace_wall_ms": reference_ms,
        "perf_margin_pct": _PERF_MARGIN * 100.0,
    }
    print("KDA_SYNTHETIC_PERF=" + json.dumps(result, sort_keys=True))
    _assert_synthetic_performance(layout, median_wall_ms)
