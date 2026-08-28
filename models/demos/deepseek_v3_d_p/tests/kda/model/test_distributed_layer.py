# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end KDA sequence-parallel correctness tests."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    assert_accurate,
    assert_bit_identical,
    random_weights,
    reconstruct_convolution_at_sp_rank,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.tt_transformers.tt.ccl import TT_CCL

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]


def _to_sp_input(
    hidden: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
) -> ttnn.Tensor:
    mesh_dims = [None, None]
    mesh_dims[sp_axis] = 1
    return ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(mesh_dims), mesh_shape=tuple(mesh_device.shape)),
    )


@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
def test_sp_layer_matches_serial_reference(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    config = KDAConfig(
        hidden_size=128,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    weights = random_weights(config)
    sequence = 1024
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(937)).to(
        torch.bfloat16
    )
    expected_output, expected_state = kda_forward_reference(hidden, weights, config)

    sp_axis = 1 - tensor_parallel_axis
    program_config = KDAProgramConfig(
        recurrence=KDARecurrenceProgramConfig(summary_group_chunks=8),
        gated_rms_output_dtype=ttnn.bfloat16,
        output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
    )
    layer = ttKDA(
        mesh_device,
        config,
        weights,
        tt_ccl=TT_CCL(mesh_device),
        sp_axis=sp_axis,
        tp_axis=tensor_parallel_axis,
        program_config=program_config,
    )
    initial_state = layer.allocate_state(batch_size=1)
    hidden_tt = _to_sp_input(hidden, mesh_device, sp_axis)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output_tt, state = layer.forward(hidden_tt, initial_state)
    assert len(output_tt.shape) == 3

    actual_output = reconstruct_sp_tp_tensor(
        output_tt,
        mesh_device,
        sp_axis,
        tensor_parallel_axis,
        tp_dim=2,
        sp_dim=1,
    )
    expected_convolution = torch.cat(
        (expected_state.q_convolution, expected_state.k_convolution, expected_state.v_convolution), dim=-1
    )
    local_heads = config.num_heads // tuple(mesh_device.shape)[tensor_parallel_axis]
    local_width = local_heads * config.head_k_dim
    sp_size = tuple(mesh_device.shape)[sp_axis]

    assert_accurate(
        expected_output,
        actual_output,
        name=f"tp_axis={tensor_parallel_axis} output",
        pcc_threshold=0.98,
    )
    for sp_rank in range(sp_size):
        actual_recurrent = reconstruct_state_at_sp_rank(
            state.recurrent, mesh_device, sp_axis, tensor_parallel_axis, sp_rank
        )
        actual_convolution = reconstruct_convolution_at_sp_rank(
            state.convolution,
            mesh_device,
            sp_axis,
            tensor_parallel_axis,
            sp_rank,
            local_width,
        )
        assert_accurate(
            expected_state.recurrent,
            actual_recurrent,
            name=f"tp_axis={tensor_parallel_axis} sp_rank={sp_rank} recurrent",
            pcc_threshold=0.98,
        )
        assert_accurate(
            expected_convolution,
            actual_convolution,
            name=f"tp_axis={tensor_parallel_axis} sp_rank={sp_rank} convolution",
            pcc_threshold=0.98,
        )


@pytest.mark.parametrize(
    "tensor_parallel_axis,summary_group_chunks,splits",
    [
        (1, 8, (2048, 3072)),
        (0, 8, (2048, 3072)),
        (1, 10, (2560, 2560)),
        (0, 10, (2560, 2560)),
    ],
)
def test_sp_chunked_prefill_matches_one_shot(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    summary_group_chunks: int,
    splits: tuple[int, int],
) -> None:
    config = KDAConfig(
        hidden_size=128,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    weights = random_weights(config)
    sequence = sum(splits)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(421)).to(
        torch.bfloat16
    )
    sp_axis = 1 - tensor_parallel_axis
    layer = ttKDA(
        mesh_device,
        config,
        weights,
        tt_ccl=TT_CCL(mesh_device),
        sp_axis=sp_axis,
        tp_axis=tensor_parallel_axis,
        program_config=KDAProgramConfig(
            recurrence=KDARecurrenceProgramConfig(summary_group_chunks=summary_group_chunks)
        ),
    )

    one_shot_input_state = layer.allocate_state(batch_size=1)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        one_shot_tt, one_shot_state = layer.forward(_to_sp_input(hidden, mesh_device, sp_axis), one_shot_input_state)
    one_shot = reconstruct_sp_tp_tensor(
        one_shot_tt,
        mesh_device,
        sp_axis,
        tensor_parallel_axis,
        tp_dim=2,
        sp_dim=1,
    )
    one_shot_recurrent = reconstruct_state_at_sp_rank(
        one_shot_state.recurrent, mesh_device, sp_axis, tensor_parallel_axis, sp_rank=0
    )
    local_heads = config.num_heads // tuple(mesh_device.shape)[tensor_parallel_axis]
    one_shot_convolution = reconstruct_convolution_at_sp_rank(
        one_shot_state.convolution,
        mesh_device,
        sp_axis,
        tensor_parallel_axis,
        sp_rank=0,
        local_width=local_heads * config.head_k_dim,
    )

    chunked_state = layer.allocate_state(batch_size=1)
    outputs = []
    start = 0
    with ttnn.manage_config("throw_exception_on_fallback", True):
        for split in splits:
            stop = start + split
            output_tt, chunked_state = layer.forward(
                _to_sp_input(hidden[:, start:stop], mesh_device, sp_axis), chunked_state
            )
            outputs.append(
                reconstruct_sp_tp_tensor(
                    output_tt,
                    mesh_device,
                    sp_axis,
                    tensor_parallel_axis,
                    tp_dim=2,
                    sp_dim=1,
                )
            )
            start = stop

    chunked = torch.cat(outputs, dim=1)
    chunked_recurrent = reconstruct_state_at_sp_rank(
        chunked_state.recurrent, mesh_device, sp_axis, tensor_parallel_axis, sp_rank=0
    )
    chunked_convolution = reconstruct_convolution_at_sp_rank(
        chunked_state.convolution,
        mesh_device,
        sp_axis,
        tensor_parallel_axis,
        sp_rank=0,
        local_width=local_heads * config.head_k_dim,
    )
    label = f"tp_axis={tensor_parallel_axis} group={summary_group_chunks}"
    assert_accurate(one_shot, chunked, name=f"{label} chunked output", pcc_threshold=0.98)
    assert_accurate(
        one_shot_recurrent,
        chunked_recurrent,
        name=f"{label} chunked recurrent",
        pcc_threshold=0.98,
    )
    assert_accurate(
        one_shot_convolution,
        chunked_convolution,
        name=f"{label} chunked convolution",
        pcc_threshold=0.98,
    )


@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
def test_sp_layer_determinism(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    config = KDAConfig(
        hidden_size=128,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    weights = random_weights(config)
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    sequence = sp_size * ttnn.TILE_SIZE
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(2421)).to(
        torch.bfloat16
    )
    layer = ttKDA(
        mesh_device,
        config,
        weights,
        tt_ccl=TT_CCL(mesh_device),
        sp_axis=sp_axis,
        tp_axis=tensor_parallel_axis,
        program_config=KDAProgramConfig(recurrence=KDARecurrenceProgramConfig(summary_group_chunks=1)),
    )
    hidden_tt = _to_sp_input(hidden, mesh_device, sp_axis)
    tp_size = tuple(mesh_device.shape)[tensor_parallel_axis]
    local_width = config.num_heads // tp_size * config.head_k_dim
    results = []

    for _ in range(3):
        state = layer.allocate_state(batch_size=1)
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output_tt, state = layer.forward(hidden_tt, state)
        ttnn.synchronize_device(mesh_device)
        tensors = [reconstruct_sp_tp_tensor(output_tt, mesh_device, sp_axis, tensor_parallel_axis, tp_dim=2, sp_dim=1)]
        for sp_rank in range(sp_size):
            tensors.append(
                reconstruct_state_at_sp_rank(state.recurrent, mesh_device, sp_axis, tensor_parallel_axis, sp_rank)
            )
            tensors.append(
                reconstruct_convolution_at_sp_rank(
                    state.convolution,
                    mesh_device,
                    sp_axis,
                    tensor_parallel_axis,
                    sp_rank,
                    local_width,
                )
            )
        results.append(tuple(tensors))

    for iteration, result in enumerate(results[1:], start=1):
        for tensor_index, (expected, actual) in enumerate(zip(results[0], result)):
            assert_bit_identical(expected, actual, name=f"SP layer tensor {tensor_index} iteration {iteration}")
