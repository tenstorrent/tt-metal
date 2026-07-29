# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end KDA sequence-parallel correctness tests."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
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


def _coordinate(sp_rank: int, tp_rank: int, sp_axis: int) -> tuple[int, int]:
    return (sp_rank, tp_rank) if sp_axis == 0 else (tp_rank, sp_rank)


def _shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _sp_tp_tensor(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    tp_dim: int,
    sp_dim: int,
) -> torch.Tensor:
    shards = _shards(tensor)
    rows, columns = tuple(mesh_device.shape)
    sp_size, tp_size = (rows, columns)[sp_axis], (rows, columns)[tp_axis]
    partitions = []
    for sp_rank in range(sp_size):
        tp_shards = []
        for tp_rank in range(tp_size):
            row, column = _coordinate(sp_rank, tp_rank, sp_axis)
            shard = shards[row * columns + column]
            if shard.ndim == 4:
                shard = shard.reshape(shard.shape[0], shard.shape[-2], shard.shape[-1])
            tp_shards.append(shard)
        partitions.append(torch.cat(tp_shards, dim=tp_dim))
    return torch.cat(partitions, dim=sp_dim)


def _state_at_sp_rank(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    sp_rank: int,
) -> torch.Tensor:
    shards = _shards(tensor)
    columns = tuple(mesh_device.shape)[1]
    tp_size = tuple(mesh_device.shape)[tp_axis]
    tp_shards = []
    for tp_rank in range(tp_size):
        row, column = _coordinate(sp_rank, tp_rank, sp_axis)
        tp_shards.append(shards[row * columns + column])
    return torch.cat(tp_shards, dim=1)


def _convolution_at_sp_rank(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    sp_rank: int,
    local_width: int,
) -> torch.Tensor:
    shards = _shards(tensor)
    columns = tuple(mesh_device.shape)[1]
    tp_size = tuple(mesh_device.shape)[tp_axis]
    physical = []
    for tp_rank in range(tp_size):
        row, column = _coordinate(sp_rank, tp_rank, sp_axis)
        physical.append(shards[row * columns + column])
    return torch.cat(
        tuple(
            torch.cat([shard[..., index * local_width : (index + 1) * local_width] for shard in physical], dim=-1)
            for index in range(3)
        ),
        dim=-1,
    )


def _assert_pcc(name: str, expected: torch.Tensor, actual: torch.Tensor, threshold: float = 0.98) -> None:
    passed, pcc = comp_pcc(expected, actual, pcc=threshold)
    max_abs = (expected.float() - actual.float()).abs().max().item()
    print(f"{name}: PCC={pcc:.6f}, max_abs={max_abs:.6e}")
    assert passed, f"{name} PCC {pcc:.6f} < {threshold}"


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
    layer = KimiDeltaAttention(
        mesh_device,
        config,
        weights,
        tt_ccl=TT_CCL(mesh_device),
        tensor_parallel_axis=tensor_parallel_axis,
        summary_group_chunks=8,
    )
    layer.reset_state(batch_size=1)
    hidden_tt = _to_sp_input(hidden, mesh_device, sp_axis)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output_tt = layer.forward(hidden_tt)

    actual_output = _sp_tp_tensor(
        output_tt,
        mesh_device,
        sp_axis,
        tensor_parallel_axis,
        tp_dim=2,
        sp_dim=1,
    )
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    expected_convolution = torch.cat(
        (expected_state.q_convolution, expected_state.k_convolution, expected_state.v_convolution), dim=-1
    )
    local_heads = config.num_heads // tuple(mesh_device.shape)[tensor_parallel_axis]
    local_width = local_heads * config.head_k_dim
    sp_size = tuple(mesh_device.shape)[sp_axis]

    _assert_pcc(f"tp_axis={tensor_parallel_axis} output", expected_output, actual_output)
    for sp_rank in range(sp_size):
        actual_recurrent = _state_at_sp_rank(layer.recurrent_state, mesh_device, sp_axis, tensor_parallel_axis, sp_rank)
        actual_convolution = _convolution_at_sp_rank(
            layer.convolution_state,
            mesh_device,
            sp_axis,
            tensor_parallel_axis,
            sp_rank,
            local_width,
        )
        _assert_pcc(
            f"tp_axis={tensor_parallel_axis} sp_rank={sp_rank} recurrent",
            expected_state.recurrent,
            actual_recurrent,
        )
        _assert_pcc(
            f"tp_axis={tensor_parallel_axis} sp_rank={sp_rank} convolution",
            expected_convolution,
            actual_convolution,
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
    layer = KimiDeltaAttention(
        mesh_device,
        config,
        weights,
        tt_ccl=TT_CCL(mesh_device),
        tensor_parallel_axis=tensor_parallel_axis,
        summary_group_chunks=summary_group_chunks,
    )

    layer.reset_state(batch_size=1)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        one_shot_tt = layer.forward(_to_sp_input(hidden, mesh_device, sp_axis))
    one_shot = _sp_tp_tensor(
        one_shot_tt,
        mesh_device,
        sp_axis,
        tensor_parallel_axis,
        tp_dim=2,
        sp_dim=1,
    )
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    one_shot_recurrent = _state_at_sp_rank(layer.recurrent_state, mesh_device, sp_axis, tensor_parallel_axis, sp_rank=0)
    local_heads = config.num_heads // tuple(mesh_device.shape)[tensor_parallel_axis]
    one_shot_convolution = _convolution_at_sp_rank(
        layer.convolution_state,
        mesh_device,
        sp_axis,
        tensor_parallel_axis,
        sp_rank=0,
        local_width=local_heads * config.head_k_dim,
    )

    layer.reset_state(batch_size=1)
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    layer.set_external_state(layer.recurrent_state, layer.convolution_state)
    recurrent_state = layer.recurrent_state
    convolution_state = layer.convolution_state
    outputs = []
    start = 0
    with ttnn.manage_config("throw_exception_on_fallback", True):
        for split in splits:
            stop = start + split
            output_tt = layer.forward(_to_sp_input(hidden[:, start:stop], mesh_device, sp_axis))
            outputs.append(
                _sp_tp_tensor(
                    output_tt,
                    mesh_device,
                    sp_axis,
                    tensor_parallel_axis,
                    tp_dim=2,
                    sp_dim=1,
                )
            )
            assert layer.recurrent_state is recurrent_state
            assert layer.convolution_state is convolution_state
            start = stop

    chunked = torch.cat(outputs, dim=1)
    chunked_recurrent = _state_at_sp_rank(layer.recurrent_state, mesh_device, sp_axis, tensor_parallel_axis, sp_rank=0)
    chunked_convolution = _convolution_at_sp_rank(
        layer.convolution_state,
        mesh_device,
        sp_axis,
        tensor_parallel_axis,
        sp_rank=0,
        local_width=local_heads * config.head_k_dim,
    )
    label = f"tp_axis={tensor_parallel_axis} group={summary_group_chunks}"
    _assert_pcc(f"{label} chunked output", one_shot, chunked)
    _assert_pcc(f"{label} chunked recurrent", one_shot_recurrent, chunked_recurrent)
    _assert_pcc(f"{label} chunked convolution", one_shot_convolution, chunked_convolution)
