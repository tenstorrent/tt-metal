# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Throwaway 5K correctness and trace timing harness for KDA padding ideas."""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    make_kimi_k3_device_case,
    make_synthetic_kimi_k3_test_case,
    reconstruct_convolution_at_sp_rank,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(900)]

_SEQUENCE = 5120
_REPETITIONS = 10
_SAMPLES = 5


def _state_tensors(state: KdaState) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    return state.recurrent, state.convolution


def _deallocate_result(result: tuple[ttnn.Tensor, KdaState]) -> None:
    output, state = result
    ttnn.deallocate(output)
    for tensor in _state_tensors(state):
        ttnn.deallocate(tensor)


def _trace_samples(
    mesh_device: ttnn.MeshDevice, run: Callable[[KdaState], tuple[ttnn.Tensor, KdaState]], input_state: KdaState
) -> list[float]:
    warm = run(input_state)
    ttnn.synchronize_device(mesh_device)
    _deallocate_result(warm)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    captured = run(input_state)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples = []
        for _ in range(_SAMPLES):
            start = time.perf_counter()
            for _ in range(_REPETITIONS):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - start) * 1e3 / _REPETITIONS)
        return samples
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        _deallocate_result(captured)
        for tensor in _state_tensors(input_state):
            ttnn.deallocate(tensor)


def _to_tp8(hidden: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(1, None), mesh_shape=tuple(mesh_device.shape)),
    )


def _reconstruct_result(
    result: tuple[ttnn.Tensor, KdaState], mesh_device: ttnn.MeshDevice, local_width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output, state = result
    return (
        reconstruct_sp_tp_tensor(output, mesh_device, 0, 1, tp_dim=2, sp_dim=1),
        reconstruct_state_at_sp_rank(state.recurrent, mesh_device, 0, 1, 0),
        reconstruct_convolution_at_sp_rank(state.convolution, mesh_device, 0, 1, 0, local_width),
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_padding_prototype_5k(mesh_device: ttnn.MeshDevice) -> None:
    length = int(os.environ.get("KDA_PAD_LENGTH", "4896"))
    if not 0 < length <= _SEQUENCE:
        raise ValueError(f"KDA_PAD_LENGTH must be in [1, {_SEQUENCE}], got {length}")
    if length % ttnn.TILE_SIZE != 0:
        raise ValueError(f"KDA_PAD_LENGTH must be divisible by {ttnn.TILE_SIZE}, got {length}")
    case = make_synthetic_kimi_k3_test_case(sequence=_SEQUENCE)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)

    baseline_samples = _trace_samples(
        mesh_device,
        lambda state: layer.forward(hidden, state),
        layer.allocate_state(),
    )
    padded_samples = _trace_samples(
        mesh_device,
        lambda state: layer.forward(hidden, state, length),
        layer.allocate_state(),
    )

    trimmed = _to_tp8(case.hidden[:, :length], mesh_device)
    baseline_input_state = layer.allocate_state()
    padded_input_state = layer.allocate_state()
    baseline_result = layer.forward(trimmed, baseline_input_state)
    padded_result = layer.forward(hidden, padded_input_state, length)
    local_width = case.config.num_heads // 8 * case.config.head_k_dim
    expected = _reconstruct_result(baseline_result, mesh_device, local_width)
    actual = _reconstruct_result(padded_result, mesh_device, local_width)
    names = ("output", "recurrent", "convolution")
    pcc = {
        name: assert_accurate(reference, observed, name=f"5K trimmed {name}", pcc_threshold=0.999)
        for name, reference, observed in zip(names, expected, actual)
    }
    zero_padded_host = case.hidden.clone()
    zero_padded_host[:, length:] = 0
    invariant_input_state = layer.allocate_state()
    invariant_result = layer.forward(_to_tp8(zero_padded_host, mesh_device), invariant_input_state, length)
    invariant = _reconstruct_result(invariant_result, mesh_device, local_width)
    for name, reference, observed in zip(names, actual, invariant):
        assert_bit_identical(reference, observed, name=f"5K padding-invariant {name}")
    for result in (baseline_result, padded_result, invariant_result):
        _deallocate_result(result)
    for state in (baseline_input_state, padded_input_state, invariant_input_state):
        for tensor in _state_tensors(state):
            ttnn.deallocate(tensor)

    result = {
        "idea": os.environ.get("KDA_PAD_IDEA", "mask"),
        "physical_sequence": _SEQUENCE,
        "length": length,
        "padding": _SEQUENCE - length,
        "baseline_samples_ms": baseline_samples,
        "baseline_median_ms": statistics.median(baseline_samples),
        "prototype_samples_ms": padded_samples,
        "prototype_median_ms": statistics.median(padded_samples),
        "ratio": statistics.median(padded_samples) / statistics.median(baseline_samples),
        "pcc": pcc,
    }
    print("KDA_PADDING_PROTOTYPE=" + json.dumps(result, sort_keys=True))
