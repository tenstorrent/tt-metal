# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Eight-device whole-head KDA weight-layout tests."""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.experimental.kimi_delta_attention.tt.weights import load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def test_tp_weight_layout(mesh_device: ttnn.MeshDevice) -> None:
    config = KDAConfig(
        hidden_size=64,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    state_dict = random_weights(config)
    weights = load_kda_weights(mesh_device, config, state_dict)

    assert weights.tensor_parallel_size == 8
    qkv_shards = _host_shards(weights.qkv_projection)
    auxiliary_shards = _host_shards(weights.auxiliary_projection)
    output_shards = _host_shards(weights.output_projection)
    tap_shards = _host_shards(weights.convolution_taps[0])

    for device_index in range(8):
        head_start = device_index * config.head_k_dim
        head_end = head_start + config.head_k_dim
        expected_qkv = torch.cat(
            (
                state_dict["q_proj.weight"][head_start:head_end],
                state_dict["k_proj.weight"][head_start:head_end],
                state_dict["v_proj.weight"][head_start:head_end],
            ),
            dim=0,
        ).T
        expected_auxiliary = torch.cat(
            (
                state_dict["f_a_proj.weight"],
                state_dict["g_a_proj.weight"],
                state_dict["b_proj.weight"][device_index : device_index + 1],
            ),
            dim=0,
        ).T
        expected_output = state_dict["o_proj.weight"][:, head_start:head_end].T
        expected_tap = torch.cat(
            (
                state_dict["q_conv1d.weight"][head_start:head_end, 0, 0],
                state_dict["k_conv1d.weight"][head_start:head_end, 0, 0],
                state_dict["v_conv1d.weight"][head_start:head_end, 0, 0],
            )
        ).reshape(1, 1, -1)

        torch.testing.assert_close(qkv_shards[device_index], expected_qkv.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(
            auxiliary_shards[device_index], expected_auxiliary.to(torch.bfloat16), rtol=0, atol=0
        )
        torch.testing.assert_close(output_shards[device_index], expected_output.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(tap_shards[device_index], expected_tap.to(torch.bfloat16), rtol=0, atol=0)


def test_tp_layer_pcc(mesh_device: ttnn.MeshDevice) -> None:
    config = KDAConfig(
        hidden_size=256,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    state_dict = random_weights(config)
    hidden = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(911)).to(torch.bfloat16)
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)

    layer = KimiDeltaAttention(mesh_device, config, state_dict, tt_ccl=TT_CCL(mesh_device))
    layer.reset_state(batch_size=1)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output = layer.forward(hidden_tt, mode="chunk")

    actual_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    recurrent_shards = _host_shards(layer.recurrent_state)
    convolution_shards = _host_shards(layer.convolution_state)
    actual_recurrent = torch.cat(recurrent_shards, dim=1)
    local_width = config.head_k_dim
    actual_convolution = torch.cat(
        (
            torch.cat([shard[..., :local_width] for shard in convolution_shards], dim=-1),
            torch.cat([shard[..., local_width : 2 * local_width] for shard in convolution_shards], dim=-1),
            torch.cat([shard[..., 2 * local_width :] for shard in convolution_shards], dim=-1),
        ),
        dim=-1,
    )
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    )

    for name, golden, actual in (
        ("output", golden_output, actual_output),
        ("recurrent state", golden_state.recurrent, actual_recurrent),
        ("convolution state", golden_convolution, actual_convolution),
    ):
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        print(f"TP=8 {name}: PCC={pcc:.6f}")
        assert passed, f"TP=8 {name} PCC {pcc:.6f} < 0.98"
