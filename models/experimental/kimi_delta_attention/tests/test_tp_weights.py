# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Eight-device whole-head KDA weight-layout tests."""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.blackhole.qwen36.tt.tp_common import matmul_reduce_scatter_prefill
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.experimental.kimi_delta_attention.tt.weights import load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(1, 8), (2, 4)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def test_tp_weight_layout(mesh_device: ttnn.MeshDevice) -> None:
    if tuple(mesh_device.shape) != (1, 8):
        pytest.skip("TP=8 layout case requires a 1x8 mesh")
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
    input_shards = _host_shards(weights.input_projection)
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
        output_gate_projection = state_dict["g_b_proj.weight"].reshape(
            config.num_heads, config.head_v_dim, config.head_v_dim
        )
        output_gate_direct = torch.matmul(output_gate_projection, state_dict["g_a_proj.weight"]).reshape(
            config.v_dim, config.hidden_size
        )
        value_start = device_index * config.head_v_dim
        value_end = value_start + config.head_v_dim
        expected_auxiliary = torch.cat(
            (
                state_dict["f_a_proj.weight"],
                output_gate_direct[value_start:value_end],
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

        expected_input = torch.cat((expected_qkv, expected_auxiliary), dim=-1)
        torch.testing.assert_close(input_shards[device_index], expected_input.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(output_shards[device_index], expected_output.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(tap_shards[device_index], expected_tap.to(torch.bfloat16), rtol=0, atol=0)


def test_tp_layer_pcc(mesh_device: ttnn.MeshDevice) -> None:
    if tuple(mesh_device.shape) != (1, 8):
        pytest.skip("TP=8 layer case requires a 1x8 mesh")
    config = KDAConfig(
        hidden_size=256,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=256,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    state_dict = random_weights(config)
    sequence = int(os.getenv("KDA_TP_TEST_SEQ", "32"))
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(911)).to(
        torch.bfloat16
    )
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
        output = layer.forward(hidden_tt)

    actual_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    recurrent_shards = _host_shards(layer.recurrent_state)
    convolution_shards = _host_shards(layer.convolution_state)
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
    )

    results = []
    for name, golden, actual in (
        ("output", golden_output, actual_output),
        ("recurrent state", golden_state.recurrent, actual_recurrent),
        ("convolution state", golden_convolution, actual_convolution),
    ):
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        print(f"TP=8 {name}: PCC={pcc:.6f}")
        results.append((name, passed, pcc))
    for name, passed, pcc in results:
        assert passed, f"TP=8 {name} PCC {pcc:.6f} < 0.98"


@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
def test_2d_tp_weight_and_output_placement(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    if tuple(mesh_device.shape) != (2, 4):
        pytest.skip("2D placement case requires a 2x4 mesh")
    config = KDAConfig(
        hidden_size=256,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=256,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    state_dict = random_weights(config)
    tensor_parallel_size = tuple(mesh_device.shape)[tensor_parallel_axis]
    weights = load_kda_weights(
        mesh_device,
        config,
        state_dict,
        tensor_parallel_axis=tensor_parallel_axis,
    )
    assert weights.tensor_parallel_size == tensor_parallel_size

    output_weight_shards = _host_shards(weights.output_projection)
    for physical_index, actual_weight in enumerate(output_weight_shards):
        row, column = divmod(physical_index, 4)
        tp_rank = (row, column)[tensor_parallel_axis]
        head_start = tp_rank * (config.num_heads // tensor_parallel_size) * config.head_v_dim
        head_end = head_start + (config.num_heads // tensor_parallel_size) * config.head_v_dim
        expected_weight = state_dict["o_proj.weight"][:, head_start:head_end].T.to(torch.bfloat16)
        torch.testing.assert_close(actual_weight, expected_weight, rtol=0, atol=0)

    sequence = 64
    value = torch.randn(1, sequence, config.v_dim, generator=torch.Generator().manual_seed(817), dtype=torch.bfloat16)
    golden_output = value @ state_dict["o_proj.weight"].T.to(torch.bfloat16)
    mesh_dims = [None, None]
    mesh_dims[tensor_parallel_axis] = 2
    value_tt = ttnn.from_torch(
        value,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(mesh_dims), mesh_shape=tuple(mesh_device.shape)),
    )
    assert (
        ttnn.get_usable_topology(value_tt, topology=ttnn.Topology.Ring, cluster_axis=tensor_parallel_axis)
        == ttnn.Topology.Linear
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    output = matmul_reduce_scatter_prefill(
        value_tt,
        weights.output_projection,
        TT_CCL(mesh_device),
        compute_config,
        ttnn.Topology.Ring,
        tensor_parallel_size,
        ttnn.bfloat16,
        cluster_axis=tensor_parallel_axis,
    )

    output_shards = _host_shards(output)
    hidden_per_rank = config.hidden_size // tensor_parallel_size
    for physical_index, actual_output in enumerate(output_shards):
        row, column = divmod(physical_index, 4)
        tp_rank = (row, column)[tensor_parallel_axis]
        expected_output = golden_output[..., tp_rank * hidden_per_rank : (tp_rank + 1) * hidden_per_rank]
        actual_output = actual_output.reshape_as(expected_output)
        passed, pcc = comp_pcc(expected_output, actual_output, pcc=0.98)
        max_abs = (expected_output.float() - actual_output.float()).abs().max().item()
        print(
            f"tp_axis={tensor_parallel_axis} device={physical_index} output: " f"PCC={pcc:.6f}, max_abs={max_abs:.6e}"
        )
        assert passed, f"tp_axis={tensor_parallel_axis} device={physical_index} output PCC {pcc:.6f} < 0.98"
