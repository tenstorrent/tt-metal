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
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric_1d_device_params,
    torus_xy_device_params,
    tp_axis_is_wrapped,
)
from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import load_or_compute_cpu_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    LayerNrmseThresholds,
    check_kimi_k3_accuracy,
    deallocate_state,
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
    make_synthetic_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged
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
_QUIETBOX_PERF_SKU = "bh_quietbox"
_PERF_MARGIN = 0.03
_PERF_REFERENCE_MS = {
    _PERF_SKU: {
        "SP1xTP8": 7.379,
        "SP2xTP4": 7.436,
        "SP4xTP2": 6.755,
    },
}
# Device time of one traced forward: the sum over programs of each program's slowest chip, as the K3 MLA perf test
# gates it. Host wall time on a QuietBox varies by several percent between replays; device time does not.
_DEVICE_PERF_REFERENCE_MS = {
    _PERF_SKU: {
        "SP4xTP2": 2.075,
    },
    _QUIETBOX_PERF_SKU: {
        "SP2xTP2": 2.045,
    },
}
_DEVICE_TIME_SAMPLES = 5
# Overrides the chunk's start position for investigation; the performance gates are skipped when it is set.
_ACTUAL_START_OVERRIDE = os.environ.get("KDA_PERF_ACTUAL_START")
# Galaxy target: a full chunk whose start falls mid-device. A resumed request starts at an arbitrary aligned
# prefix, and every full chunk of it then wraps onto its first rank, which adds the final-carry barrier and
# broadcast that a rank-aligned start skips. Production always passes actual_end, so the gate does too.
_GALAXY_PERF_REFERENCE_MS = 2.015
_GALAXY_ROWS_PER_CHIP = 640
_GALAXY_HEADS_PER_CHIP = 24
_GALAXY_ACTUAL_START = 4 * _GALAXY_ROWS_PER_CHIP + 320
# QuietBox SP2xTP2 runs Galaxy's per-chip workload: 640 rows and 24 heads per chip, so the chunk shrinks to 1280
# tokens and the synthetic config halves the global head count. Every per-chip shape, L1 placement and program
# config then matches Galaxy, and the chunk starts 320 rows into a device as on Galaxy. The SP chain is 2 ranks
# instead of 8, and the 11x10 grid splits each head's values over 2 cores instead of 4.
_QUIETBOX_SEQUENCE = 2 * _GALAXY_ROWS_PER_CHIP
_QUIETBOX_NUM_HEADS = 2 * _GALAXY_HEADS_PER_CHIP
_QUIETBOX_ACTUAL_START = _GALAXY_ROWS_PER_CHIP + 320
# LoudBox SP4xTP2 runs the same per-chip workload on a 2560-token chunk. The SP chain is 4 ranks and the TP
# reduce-scatter 2; the 11x10 grid splits each head's values over 2 cores as on QuietBox.
_LOUDBOX_SEQUENCE = 4 * _GALAXY_ROWS_PER_CHIP
_LOUDBOX_NUM_HEADS = 2 * _GALAXY_HEADS_PER_CHIP
_LOUDBOX_ACTUAL_START = 2 * _GALAXY_ROWS_PER_CHIP + 320
# Like the model's prefill tests, give CCL semaphores their own L1_SMALL region: allocated in L1 on the first call,
# they would land between the chunk terms and fragment L1, so the traced call would plan ops on less free L1 than
# the warm-up compiled for.
_L1_SMALL_SIZE = 1152


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
    references = _PERF_REFERENCE_MS.get(os.environ.get("KDA_PERF_SKU"), {})
    if layout not in references:
        skus = sorted(sku for sku, sku_references in _PERF_REFERENCE_MS.items() if layout in sku_references)
        raise ValueError(f"set KDA_PERF_SKU to one of {skus} to opt in to this hardware-specific performance gate")
    return references[layout]


def _device_perf_reference_ms(layout: str) -> float | None:
    """Return the device-time reference for a device-gated layout, or None for layouts gated on host wall time."""
    skus = sorted(sku for sku, references in _DEVICE_PERF_REFERENCE_MS.items() if layout in references)
    if not skus:
        return None
    sku = os.environ.get("KDA_PERF_SKU")
    if sku not in skus:
        raise ValueError(f"set KDA_PERF_SKU to one of {skus} to opt in to this hardware-specific performance gate")
    return _DEVICE_PERF_REFERENCE_MS[sku][layout]


def _allocate_state(layer: ttKDA) -> KdaState:
    return layer.allocate_state(batch_size=1)


def _synthetic_perf_reference_ms(layout: str) -> float:
    if layout == "SP8xTP4":
        return _GALAXY_PERF_REFERENCE_MS
    return _perf_reference_ms(layout)


def _assert_within_margin(what: str, measured_ms: float, reference_ms: float) -> None:
    lower = reference_ms * (1.0 - _PERF_MARGIN)
    upper = reference_ms * (1.0 + _PERF_MARGIN)
    assert lower <= measured_ms <= upper, (
        f"{what} {measured_ms:.3f} ms is outside performance range "
        f"[{lower:.3f}, {upper:.3f}] ms "
        f"(reference {reference_ms:.3f} ms ± {_PERF_MARGIN:.0%})"
    )


def _assert_synthetic_performance(layout: str, median_wall_ms: float) -> None:
    _assert_within_margin(f"{layout} median trace wall", median_wall_ms, _synthetic_perf_reference_ms(layout))


@pytest.mark.parametrize(
    "layout,sku,reference",
    [
        ("SP4xTP2", _PERF_SKU, _device_perf_reference_ms),
        ("SP2xTP2", _QUIETBOX_PERF_SKU, _device_perf_reference_ms),
        ("SP8xTP4", _PERF_SKU, _synthetic_perf_reference_ms),
    ],
)
def test_synthetic_performance_uses_two_sided_margin(layout, sku, reference, monkeypatch, expect_error) -> None:
    monkeypatch.setenv("KDA_PERF_SKU", sku)
    reference_ms = reference(layout)
    _assert_within_margin(layout, reference_ms, reference_ms)
    with expect_error(AssertionError, "outside performance range"):
        _assert_within_margin(layout, reference_ms * 0.96, reference_ms)
    with expect_error(AssertionError, "outside performance range"):
        _assert_within_margin(layout, reference_ms * 1.04, reference_ms)


def _trace_device_samples_ms(mesh_device: ttnn.MeshDevice, trace_id: int) -> list[float]:
    """Device time of each of _DEVICE_TIME_SAMPLES replays: the sum over programs of each program's slowest chip."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("realtime profiler is inactive; the KDA device-time gate needs it")
    samples_ms = []
    for _ in range(_DEVICE_TIME_SAMPLES):
        _, programs = profile_realtime_program_merged(
            mesh_device, lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        )
        samples_ms.append(sum(program["duration_ns"] for program in programs.values()) / 1e6)
    return samples_ms


def _trace_wall_samples_ms(
    mesh_device: ttnn.MeshDevice,
    layer: ttKDA,
    hidden: ttnn.Tensor,
    repetitions: int,
    validate_first_replay: Callable[[KdaState, ttnn.Tensor], dict[str, float]] | None = None,
    actual_start: int = 0,
    actual_end: int | None = None,
    on_trace: Callable[[int], None] | None = None,
) -> tuple[list[float], dict[str, float] | None]:
    state = None
    warm_output = None
    warm_state = None
    trace_id = None
    output = None
    next_state = None
    actual_start_tt = make_actual_start(mesh_device, actual_start)
    actual_end_tt = None if actual_end is None else make_actual_start(mesh_device, actual_end)
    try:
        state = _allocate_state(layer)
        warm_output, warm_state = layer.forward(hidden, state, actual_start_tt, actual_end_tt)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(warm_output)
        warm_output = None
        deallocate_state(warm_state)
        warm_state = None

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output, next_state = layer.forward(hidden, state, actual_start_tt, actual_end_tt)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        validation = (
            validate_first_replay(
                next_state, output, nrmse=LayerNrmseThresholds(output=0.03, recurrent=0.04, convolution=0.007)
            )
            if validate_first_replay is not None
            else None
        )

        samples_ms = []
        for _ in range(_TIMING_SAMPLES):
            start = time.perf_counter()
            for _ in range(repetitions):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples_ms.append((time.perf_counter() - start) * 1e3 / repetitions)
        if on_trace is not None:
            on_trace(trace_id)
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
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "l1_small_size": _L1_SMALL_SIZE,
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
            nrmse=LayerNrmseThresholds(output=0.03, recurrent=0.04, convolution=0.007),
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
        pytest.param((2, 4), 0, fabric_1d_device_params(l1_small_size=_L1_SMALL_SIZE), id="SP4xTP2-fabric-1d"),
        pytest.param((2, 2), 1, fabric_1d_device_params(l1_small_size=_L1_SMALL_SIZE), id="SP2xTP2-fabric-1d"),
        pytest.param(
            (8, 4),
            1,
            torus_xy_device_params(l1_small_size=_L1_SMALL_SIZE),
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
    """Gate checkpoint-free production K3 latency on LoudBox, QuietBox and Galaxy."""
    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    layout = f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}"
    device_reference_ms = _device_perf_reference_ms(layout)
    reference_ms = _synthetic_perf_reference_ms(layout) if device_reference_ms is None else None
    sequence, num_heads, actual_start = {
        "SP8xTP4": (_SEQUENCE, None, _GALAXY_ACTUAL_START),
        "SP4xTP2": (_LOUDBOX_SEQUENCE, _LOUDBOX_NUM_HEADS, _LOUDBOX_ACTUAL_START),
        "SP2xTP2": (_QUIETBOX_SEQUENCE, _QUIETBOX_NUM_HEADS, _QUIETBOX_ACTUAL_START),
    }.get(layout, (_SEQUENCE, None, 0))
    if _ACTUAL_START_OVERRIDE is not None:
        actual_start = int(_ACTUAL_START_OVERRIDE)
    actual_end = None if actual_start == 0 else actual_start + sequence
    case = make_synthetic_kimi_k3_test_case(sequence=sequence, num_heads=num_heads)
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )
    if layout in ("SP8xTP4", "SP4xTP2", "SP2xTP2"):
        assert (layer.active_seq_len_local, layer.config.num_heads) == (
            _GALAXY_ROWS_PER_CHIP,
            _GALAXY_HEADS_PER_CHIP,
        ), f"{layout} must run Galaxy's per-chip workload"
    device_samples_ms = []
    samples_ms, _ = _trace_wall_samples_ms(
        mesh_device,
        layer,
        hidden_tt,
        _REPETITIONS,
        actual_start=actual_start,
        actual_end=actual_end,
        on_trace=(
            None
            if device_reference_ms is None
            else lambda trace_id: device_samples_ms.extend(_trace_device_samples_ms(mesh_device, trace_id))
        ),
    )
    median_wall_ms = statistics.median(samples_ms)
    result = {
        "fabric_config": ttnn.get_fabric_config().name,
        "layout": layout,
        "sequence": sequence,
        "num_heads": layer.config.num_heads * layer.tensor_parallel_size,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "weights": "deterministic synthetic",
        "repetitions": _REPETITIONS,
        "trace_wall_samples_ms": samples_ms,
        "median_trace_wall_ms": median_wall_ms,
        "timing_sample_count": _TIMING_SAMPLES,
        "reference_trace_wall_ms": reference_ms,
        "perf_margin_pct": _PERF_MARGIN * 100.0,
    }
    if device_reference_ms is not None:
        result |= {
            "device_samples_ms": device_samples_ms,
            "median_device_ms": statistics.median(device_samples_ms),
            "reference_device_ms": device_reference_ms,
        }
    print("KDA_SYNTHETIC_PERF=" + json.dumps(result, sort_keys=True))
    if _ACTUAL_START_OVERRIDE is not None:
        pytest.skip(f"KDA_PERF_ACTUAL_START={_ACTUAL_START_OVERRIDE} overrides the gated start; not gating")
    if layout == "SP8xTP4" and not tp_axis_is_wrapped(mesh_device):
        pytest.skip(
            f"TP axis not wrapped: measured {median_wall_ms:.3f} ms on Linear TP; "
            "the Galaxy reference assumes the TP ring"
        )
    if device_reference_ms is not None:
        _assert_within_margin(f"{layout} median device time", statistics.median(device_samples_ms), device_reference_ms)
    else:
        _assert_synthetic_performance(layout, median_wall_ms)
