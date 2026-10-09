# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local real-weight and CI-synthetic performance acceptance for Kimi-K3 KDA."""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric_1d_device_params,
    torus_xy_device_params,
    tp_axis_is_wrapped,
)
from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import load_or_compute_cpu_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    check_kimi_k3_accuracy,
    deallocate_state,
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
    make_synthetic_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry, KdaStates
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
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
# LoudBox targets.
_PERF_REFERENCE_MS = {
    "SP1xTP8": 8.617,
    "SP2xTP4": 8.758,
    "SP4xTP2": 9.066,
}
_GALAXY_PERF_REFERENCE_MS = 3.121


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
            ttnn.deallocate(actual_start_tt)


def _paired_policy_samples_ms(mesh_device, control_layer, enabled_layer, hidden):
    """Bracket each policy sample with controls at the same position to account for timing drift."""
    start_tt = make_actual_start(mesh_device, 0)
    bounds = {position: make_actual_start(mesh_device, position) for position in (0, _SEQUENCE)}
    state = control_layer.allocate_state()
    traces, outputs, returned_states = {}, [], []
    try:
        warm_output, carried = control_layer.forward(hidden, state, start_tt)
        ttnn.deallocate(warm_output)
        deallocate_state(state)
        state = carried  # Every continuation sample reads a real nonzero carry.
        for layer in (control_layer, enabled_layer):
            output, replacement = layer.forward(hidden, state, start_tt)
            ttnn.deallocate(output)
            deallocate_state(replacement)
        for name, layer in (("control", control_layer), ("enabled", enabled_layer)):
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            output, replacement = layer.forward(hidden, state, start_tt)
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            traces[name] = trace
            outputs.append(output)
            returned_states.append(replacement)
        ttnn.synchronize_device(mesh_device)

        def measure(name):
            begin = time.perf_counter()
            for _ in range(_REPETITIONS):
                ttnn.execute_trace(mesh_device, traces[name], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            return (time.perf_counter() - begin) * 1e3 / _REPETITIONS

        samples = {position: [] for position in bounds}
        for _ in range(_TIMING_SAMPLES):
            for position, scalar in bounds.items():
                ttnn.copy(scalar, start_tt)
                ttnn.synchronize_device(mesh_device)
                before = measure("control")
                enabled = measure("enabled")
                after = measure("control")
                samples[position].append(
                    {
                        "control_before_ms": before,
                        "enabled_ms": enabled,
                        "control_after_ms": after,
                        "ratio_to_control": enabled / ((before + after) / 2),
                    }
                )
        return samples
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh_device, trace)
        for output in outputs:
            ttnn.deallocate(output)
        for replacement in returned_states:
            deallocate_state(replacement)
        deallocate_state(state)
        for tensor in (start_tt, *bounds.values()):
            ttnn.deallocate(tensor)


def _request_wall_samples_ms(mesh_device, control_layer, enabled_layer, hidden):
    """Interleave request-head samples to avoid confounding the reset comparison with clock drift."""
    cache = KdaStateCache({0: control_layer})
    geometry = KdaContractGeometry.from_kda_config(
        replace(control_layer.config, num_heads=control_layer.config.num_heads * control_layer.tensor_parallel_size),
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=control_layer.sequence_parallel_axis,
        tp_axis=control_layer.tensor_parallel_axis,
    )
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=(0,), num_slots=1)
    cache.bind_slabs(slabs)
    # All persistent storage exists before either trace. This test-only zero pair emulates the old reset.
    zeros = control_layer.allocate_state()
    start_tt = make_actual_start(mesh_device, 0)
    traces, outputs = {}, []

    def forward(layer):
        output, state = layer.forward(hidden, cache.read(0), start_tt)
        cache.commit(0, state)
        return output

    def reset_control():
        current = cache.read(0)
        ttnn.copy(zeros.recurrent, current.recurrent)
        ttnn.copy(zeros.convolution, current.convolution)
        slabs.export_layer(zeros, 0, 0)

    try:
        for layer in (control_layer, enabled_layer):
            reset_control()
            ttnn.deallocate(forward(layer))
        for name, layer in (("control", control_layer), ("enabled", enabled_layer)):
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            outputs.append(forward(layer))
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            traces[name] = trace
        ttnn.synchronize_device(mesh_device)

        def measure(name):
            begin = time.perf_counter()
            for _ in range(_REPETITIONS):
                if name == "control":
                    reset_control()
                ttnn.execute_trace(mesh_device, traces[name], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            return (time.perf_counter() - begin) * 1e3 / _REPETITIONS

        samples = {name: [] for name in traces}
        for _ in range(_TIMING_SAMPLES):
            before = measure("control")
            enabled = measure("enabled")
            after = measure("control")
            samples["control"].append((before + after) / 2)
            samples["enabled"].append(enabled)
        return samples["control"], samples["enabled"]
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh_device, trace)
        for output in outputs:
            ttnn.deallocate(output)
        deallocate_state(zeros)
        cache.deallocate()
        for tensor in (slabs.recurrent, slabs.convolution, start_tt):
            ttnn.deallocate(tensor)


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
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            },
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
        pytest.param((2, 4), 1, fabric_1d_device_params(), id="SP2xTP4-fabric-1d"),
        pytest.param(
            (8, 4),
            1,
            torus_xy_device_params(),
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
    if layout == "SP8xTP4" and not tp_axis_is_wrapped(mesh_device):
        pytest.skip(
            f"TP axis not wrapped: measured {median_wall_ms:.3f} ms on Linear TP; "
            "the Galaxy reference assumes the TP ring"
        )
    _assert_synthetic_performance(layout, median_wall_ms)


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [pytest.param((2, 4), 1, fabric_1d_device_params(), id="LB-SP2xTP4-fabric-1d")],
    indirect=["mesh_device", "device_params"],
)
def test_request_initialization_perf_loudbox(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    device_params: dict,
) -> None:
    """Compare request initialization with a same-run control on LoudBox only."""
    layout = "SP2xTP4"
    _perf_reference_ms(layout)  # Require the explicit LoudBox SKU opt-in, without changing CI baselines.
    case = make_synthetic_kimi_k3_test_case(sequence=_SEQUENCE)
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )
    # Keep a same-run generic control; hardware baseline drift must not hide policy overhead.
    enabled_config = replace(
        kimi_k3_program_config(active_seq_len_local=layer.active_seq_len_local, tp_ccl_topology=layer.tp_ccl_topology),
        zero_initial_state_on_start=True,
    )
    enabled_layer, enabled_hidden = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        program_config=enabled_config,
        weights=layer.weights,
        cache_weights=False,
    )
    policy_samples = _paired_policy_samples_ms(mesh_device, layer, enabled_layer, hidden_tt)
    for position, samples in policy_samples.items():
        ratio = statistics.median(sample["ratio_to_control"] for sample in samples)
        print(
            "KDA_REQUEST_POLICY_PERF="
            + json.dumps(
                {
                    "layout": layout,
                    "actual_start": position,
                    "paired_samples": samples,
                    "median_ratio_to_control": ratio,
                },
                sort_keys=True,
            )
        )
        assert ratio <= 1 + _PERF_MARGIN
    control_requests, enabled_requests = _request_wall_samples_ms(mesh_device, layer, enabled_layer, hidden_tt)
    print(
        "KDA_REQUEST_HEAD_PERF="
        + json.dumps(
            {
                "layout": layout,
                "legacy_reset_samples_ms": control_requests,
                "device_initialization_samples_ms": enabled_requests,
                "ratio_to_legacy_reset": statistics.median(enabled_requests) / statistics.median(control_requests),
            },
            sort_keys=True,
        )
    )
    assert statistics.median(enabled_requests) <= statistics.median(control_requests) * (1 + _PERF_MARGIN)
    ttnn.deallocate(enabled_hidden)
    ttnn.deallocate(hidden_tt)
