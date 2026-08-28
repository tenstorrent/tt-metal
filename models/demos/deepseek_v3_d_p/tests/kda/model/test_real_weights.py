# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 layer-1 KDA PCC with the pinned Hugging Face weights."""

from pathlib import Path

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    check_kimi_k3_accuracy,
    kimi_k3_tensor_cache_path,
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0)],
    indirect=["mesh_device"],
    ids=["SP1xTP8", "SP2xTP4", "SP4xTP2"],
)
def test_kimi_k3_layer_1_real_weights_pcc(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_checkpoint_dir: Path,
) -> None:
    case = make_kimi_k3_test_case(kimi_k3_checkpoint_dir, sequence=128)
    golden_output, golden_state = kda_forward_reference(case.hidden, case.state_dict, case.config)
    cache_path = kimi_k3_tensor_cache_path(case.checkpoint_identity, mesh_device, tensor_parallel_axis)
    cache_prefix = f"layer_{KimiK3Config.FIRST_KDA_LAYER}.kda"
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
        mesh_device,
        case.config,
        cache_path,
        cache_prefix,
        tensor_parallel_axis=tensor_parallel_axis,
    )
    sequence_parallel_axis = 1 - tensor_parallel_axis
    local_chunks = case.hidden.shape[1] // tuple(mesh_device.shape)[sequence_parallel_axis] // ttnn.TILE_SIZE
    # Accuracy uses the shortest tile-aligned global sequence shared by all three mesh geometries.
    # Keep the production K3 tuning except for this local grouping constraint.
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
