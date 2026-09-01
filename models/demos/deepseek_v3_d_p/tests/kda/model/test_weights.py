# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Eight-device whole-head KDA weight-layout tests."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.utils import collect_mesh_accuracy_and_determinism_results, random_weights
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_equal

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
        assert_equal(
            expected_input.to(torch.bfloat16),
            input_shards[device_index],
            name=f"input weight device {device_index}",
        )
        assert_equal(
            expected_output.to(torch.bfloat16),
            output_shards[device_index],
            name=f"output weight device {device_index}",
        )
        assert_equal(
            expected_tap.to(torch.bfloat16),
            tap_shards[device_index],
            name=f"convolution tap device {device_index}",
        )


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
        assert_equal(expected_weight, actual_weight, name=f"output weight device {physical_index}")

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
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    tt_ccl = TT_CCL(mesh_device)

    def run() -> tuple[ttnn.Tensor]:
        output = ttnn.linear(
            value_tt,
            weights.output_projection,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        output = ttnn.experimental.reduce_scatter_minimal_async(
            output,
            dim=-1,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(tensor_parallel_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(tensor_parallel_axis),
            num_links=tt_ccl.get_num_links(tensor_parallel_axis),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=tensor_parallel_axis,
        )
        return (output,)

    (output,), mismatch_markers = collect_mesh_accuracy_and_determinism_results(run)
    output_shards = _host_shards(output)
    hidden_per_rank = config.hidden_size // tensor_parallel_size
    for physical_index, actual_output in enumerate(output_shards):
        row, column = divmod(physical_index, 4)
        tp_rank = (row, column)[tensor_parallel_axis]
        expected_output = golden_output[..., tp_rank * hidden_per_rank : (tp_rank + 1) * hidden_per_rank]
        actual_output = actual_output.reshape_as(expected_output)
        assert_accurate(
            expected_output,
            actual_output,
            name=f"tp_axis={tensor_parallel_axis} device={physical_index} output",
            pcc_threshold=0.999,
        )
    assert all(marker.item() == 0 for marker in mismatch_markers), "output projection is not bit-identical across runs"
