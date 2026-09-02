# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic CI and local real-weight acceptance for the Kimi-K3 KDA layer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    check_kimi_k3_accuracy,
    collect_mesh_accuracy_and_determinism_results,
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
    make_synthetic_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState

pytestmark = [run_for_blackhole(), pytest.mark.timeout(900)]

_SEQUENCE = 5120
_PCC_THRESHOLD = 0.9995


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param(
            (2, 4),
            1,
            fabric_1d_device_params(),
            id="SP2xTP4-fabric-1d",
        ),
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
def test_synthetic_kimi_k3_accuracy_and_determinism(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    device_params: dict,
) -> None:
    """Gate K3 dimensions against Torch and compare three device runs bit-for-bit."""
    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    layout = f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}"
    case = make_synthetic_kimi_k3_test_case(sequence=_SEQUENCE)
    reference_start = time.perf_counter()
    golden_output, golden_state = kda_forward_reference(case.hidden, case.state_dict, case.config)
    reference_seconds = time.perf_counter() - reference_start
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        initial_state = layer.allocate_state(batch_size=1)
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output, state = layer.forward(hidden_tt, initial_state)
        return output, state.recurrent, state.convolution

    (output, recurrent, convolution), mismatch_markers = collect_mesh_accuracy_and_determinism_results(run)
    state = KdaState(recurrent=recurrent, convolution=convolution)
    try:
        pcc = check_kimi_k3_accuracy(
            f"Synthetic Kimi-K3 T=5120 {layout}",
            case,
            golden_output,
            golden_state,
            state,
            output,
            mesh_device,
            tensor_parallel_axis,
            pcc_threshold=_PCC_THRESHOLD,
        )
        assert all(
            marker.item() == 0 for marker in mismatch_markers
        ), "synthetic Kimi-K3 outputs and states are not bit-identical across three runs"
        print(
            "KDA_SYNTHETIC_ACCURACY_DETERMINISM="
            + json.dumps(
                {
                    "layout": layout,
                    "sequence": _SEQUENCE,
                    "weights": "deterministic synthetic",
                    "reference": "independent pure-Torch FP32 CPU reference",
                    "cpu_reference_seconds": reference_seconds,
                    "pcc": pcc,
                    "determinism_repetitions": 3,
                    "bit_identical": True,
                },
                sort_keys=True,
            )
        )
    finally:
        ttnn.deallocate(output)
        ttnn.deallocate(recurrent)
        ttnn.deallocate(convolution)


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params,sequence",
    [
        pytest.param((1, 8), 1, fabric_1d_device_params(), 128, id="SP1xTP8"),
        pytest.param((2, 4), 1, fabric_1d_device_params(), 128, id="SP2xTP4"),
        pytest.param((2, 4), 0, fabric_1d_device_params(), 128, id="SP4xTP2"),
        pytest.param(
            (8, 4),
            1,
            torus_xy_device_params(),
            512,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_kimi_k3_layer_1_real_weights_accuracy(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_checkpoint_dir: Path,
    sequence: int,
) -> None:
    case = make_kimi_k3_test_case(kimi_k3_checkpoint_dir, sequence=sequence)
    golden_output, golden_state = kda_forward_reference(case.hidden, case.state_dict, case.config)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    local_chunks = case.hidden.shape[1] // tuple(mesh_device.shape)[sequence_parallel_axis] // ttnn.TILE_SIZE
    # Existing eight-device layouts retain their shortest common tile-aligned T=128; the
    # Galaxy case uses T=512 so SP8 has two local chunks. Keep production K3 tuning
    # except for this local grouping constraint.
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        summary_group_chunks=local_chunks,
    )
    state = layer.allocate_state(batch_size=1)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, state = layer.forward(hidden_tt, state)
    ttnn.synchronize_device(mesh_device)

    mesh_shape = tuple(mesh_device.shape)
    layout = f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}"
    check_kimi_k3_accuracy(
        f"Kimi-K3 layer 1 {layout}",
        case,
        golden_output,
        golden_state,
        state,
        output,
        mesh_device,
        tensor_parallel_axis,
        pcc_threshold=0.9995,
    )
