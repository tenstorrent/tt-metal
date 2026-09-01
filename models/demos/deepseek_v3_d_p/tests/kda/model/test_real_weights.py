# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 layer-1 KDA PCC with the pinned Hugging Face weights."""

from pathlib import Path

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import KIMI_K3_FIRST_KDA_LAYER
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    check_kimi_k3_accuracy,
    kimi_k3_tensor_cache_path,
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights

pytestmark = [run_for_blackhole()]


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params,sequence",
    [
        pytest.param((1, 8), 1, fabric_1d_device_params(l1_small_size=24576), 128, id="SP1xTP8"),
        pytest.param((2, 4), 1, fabric_1d_device_params(l1_small_size=24576), 128, id="SP2xTP4"),
        pytest.param((2, 4), 0, fabric_1d_device_params(l1_small_size=24576), 128, id="SP4xTP2"),
        pytest.param(
            (8, 4),
            1,
            torus_xy_device_params(l1_small_size=24576),
            512,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_kimi_k3_layer_1_real_weights_pcc(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_checkpoint_dir: Path,
    sequence: int,
) -> None:
    case = make_kimi_k3_test_case(kimi_k3_checkpoint_dir, sequence=sequence)
    golden_output, golden_state = kda_forward_reference(case.hidden, case.state_dict, case.config)
    cache_path = kimi_k3_tensor_cache_path(case.weights_identity, mesh_device, tensor_parallel_axis)
    cache_prefix = f"layer_{KIMI_K3_FIRST_KDA_LAYER}.kda"
    if not KDAWeights.check_cache_complete(
        cache_path,
        cache_prefix,
        case.config,
        mesh_device,
        tensor_parallel_axis=tensor_parallel_axis,
    ):
        KDAWeights.build_ttnn_cache(
            case.state_dict,
            cache_path,
            cache_prefix,
            case.config,
            mesh_device,
            tensor_parallel_axis=tensor_parallel_axis,
        )
    assert KDAWeights.check_cache_complete(
        cache_path,
        cache_prefix,
        case.config,
        mesh_device,
        tensor_parallel_axis=tensor_parallel_axis,
    )
    cached_weights = KDAWeights.from_cache(
        cache_path,
        cache_prefix,
        case.config,
        mesh_device,
        tensor_parallel_axis=tensor_parallel_axis,
    )
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
        weights=cached_weights,
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
