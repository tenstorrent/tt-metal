# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device weight-placement contracts for the KDA layer."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_equal

pytestmark = run_for_blackhole()


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _tp_rank(physical_index: int, mesh_columns: int, tensor_parallel_axis: int) -> int:
    row, column = divmod(physical_index, mesh_columns)
    return (row, column)[tensor_parallel_axis]


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [
        pytest.param((1, 8), 1, id="tp8-1d"),
        pytest.param((2, 4), 0, id="tp2-axis0"),
        pytest.param((2, 4), 1, id="tp4-axis1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_device_weight_placement(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    config = KDAConfig(
        hidden_size=64,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    state_dict = random_weights(config)
    weights = load_kda_weights(
        mesh_device,
        config,
        state_dict,
        tensor_parallel_axis=tensor_parallel_axis,
    )
    tensor_parallel_size = tuple(mesh_device.shape)[tensor_parallel_axis]
    local_heads = config.num_heads // tensor_parallel_size
    gate_projection = state_dict["g_b_proj.weight"].reshape(config.num_heads, config.head_v_dim, config.head_v_dim)
    direct_gate = torch.matmul(gate_projection, state_dict["g_a_proj.weight"]).reshape(config.v_dim, config.hidden_size)
    decay_scale = -state_dict["A_log"].float().exp()
    decay_scale = decay_scale.expand(-1, -1, -1, config.head_k_dim).reshape(1, 1, config.q_dim)
    decay_bias = state_dict["dt_bias"].reshape(1, 1, config.q_dim)

    assert weights.tensor_parallel_size == tensor_parallel_size
    assert weights.tensor_parallel_axis == tensor_parallel_axis
    fields = {
        "input": _host_shards(weights.input_projection),
        "decay_output": _host_shards(weights.decay_output_projection),
        "output": _host_shards(weights.output_projection),
        "decay_scale": _host_shards(weights.decay_scale_flat),
        "decay_bias": _host_shards(weights.decay_bias_flat),
        "norm": _host_shards(weights.norm),
        "tap": _host_shards(weights.convolution_taps[0]),
    }

    for physical_index in range(len(fields["input"])):
        tp_rank = (
            _tp_rank(physical_index, tuple(mesh_device.shape)[1], tensor_parallel_axis)
            if tuple(mesh_device.shape)[0] > 1
            else physical_index
        )
        head_start = tp_rank * local_heads
        head_stop = head_start + local_heads
        key_start, key_stop = head_start * config.head_k_dim, head_stop * config.head_k_dim
        value_start, value_stop = head_start * config.head_v_dim, head_stop * config.head_v_dim
        expected_qkv = torch.cat(
            (
                state_dict["q_proj.weight"][key_start:key_stop],
                state_dict["k_proj.weight"][key_start:key_stop],
                state_dict["v_proj.weight"][value_start:value_stop],
            ),
            dim=0,
        ).T
        expected_auxiliary = torch.cat(
            (
                state_dict["f_a_proj.weight"],
                direct_gate[value_start:value_stop],
                state_dict["b_proj.weight"][head_start:head_stop],
            ),
            dim=0,
        ).T
        expected_tap = torch.cat(
            (
                state_dict["q_conv1d.weight"][key_start:key_stop, 0, 0],
                state_dict["k_conv1d.weight"][key_start:key_stop, 0, 0],
                state_dict["v_conv1d.weight"][value_start:value_stop, 0, 0],
            )
        ).reshape(1, 1, -1)
        expected = {
            "input": torch.cat((expected_qkv, expected_auxiliary), dim=-1),
            "decay_output": state_dict["f_b_proj.weight"].T[:, key_start:key_stop],
            "output": state_dict["o_proj.weight"][:, value_start:value_stop].T,
            "decay_scale": decay_scale[..., key_start:key_stop],
            "decay_bias": decay_bias[..., key_start:key_stop],
            "norm": state_dict["o_norm.weight"],
            "tap": expected_tap,
        }
        for name, expected_tensor in expected.items():
            assert_equal(
                expected_tensor.to(torch.bfloat16),
                fields[name][physical_index],
                name=f"{name} weight device {physical_index}",
            )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_tp_layer_with_nonsquare_state_matches_reference(mesh_device: ttnn.MeshDevice) -> None:
    config = KDAConfig(
        hidden_size=256,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=256,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    state_dict = random_weights(config)
    sequence = 32
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(911)).to(
        torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)

    layer = ttKDA(mesh_device, config, state_dict, tt_ccl=TT_CCL(mesh_device))
    initial_state = layer.allocate_state(batch_size=1)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, state = layer.forward(hidden_tt, initial_state)

    actual_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    recurrent_shards = _host_shards(state.recurrent)
    convolution_shards = _host_shards(state.convolution)
    actual_recurrent = torch.cat(recurrent_shards, dim=1)
    local_key_width = config.head_k_dim
    local_value_width = config.head_v_dim
    actual_convolution = torch.cat(
        (
            torch.cat([shard[..., :local_key_width] for shard in convolution_shards], dim=-1),
            torch.cat([shard[..., local_key_width : 2 * local_key_width] for shard in convolution_shards], dim=-1),
            torch.cat(
                [
                    shard[..., 2 * local_key_width : 2 * local_key_width + local_value_width]
                    for shard in convolution_shards
                ],
                dim=-1,
            ),
        ),
        dim=-1,
    )
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    ).to(torch.bfloat16)

    for name, golden, actual in (
        ("output", golden_output, actual_output),
        ("recurrent state", golden_state.recurrent, actual_recurrent),
        ("convolution state", golden_convolution, actual_convolution),
    ):
        assert_accurate(golden, actual, name=f"TP=8 {name}", pcc_threshold=0.999)
