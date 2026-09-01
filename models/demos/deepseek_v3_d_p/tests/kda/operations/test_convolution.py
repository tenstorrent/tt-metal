# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Blackhole correctness tests for the 2D-mesh KDA convolution halo."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.kda.utils import collect_mesh_accuracy_and_determinism_results
from models.demos.deepseek_v3_d_p.tt.kda.convolution import exchange_convolution_carry
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_equal

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


def _to_device(tensor: torch.Tensor, device: ttnn.MeshDevice, dims: tuple[int | None, int | None]) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=dims, mesh_shape=tuple(device.shape)),
    )


def _sp_carries(tensor: ttnn.Tensor, device: ttnn.MeshDevice, sp_axis: int, tp_axis: int) -> torch.Tensor:
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
    rows, columns = tuple(device.shape)
    sp_size, tp_size = (rows, columns)[sp_axis], (rows, columns)[tp_axis]
    partitions = []
    for sp_rank in range(sp_size):
        channel_shards = []
        for tp_rank in range(tp_size):
            row, column = _coordinate(sp_rank, tp_rank, sp_axis)
            channel_shards.append(shards[row * columns + column])
        partitions.append(torch.cat(channel_shards, dim=2))
    return torch.stack(partitions)


@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
def test_exchange_convolution_carry_preserves_causal_carries(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    tp_size = tuple(mesh_device.shape)[tensor_parallel_axis]
    batch, local_sequence, history = 1, 8, 3
    channels = tp_size * 32
    sequence = sp_size * local_sequence

    qkv = torch.arange(batch * sequence * channels, dtype=torch.float32).reshape(batch, sequence, channels)
    qkv = ((qkv.remainder(97) - 48) / 8).to(torch.bfloat16)
    external = torch.arange(batch * history * channels, dtype=torch.float32).reshape(batch, history, channels)
    external = (-(external.remainder(31) + 1) / 8).to(torch.bfloat16)

    qkv_dims = [None, None]
    qkv_dims[sp_axis], qkv_dims[tensor_parallel_axis] = 1, 2
    state_dims = [None, None]
    state_dims[tensor_parallel_axis] = 2
    qkv_tt = _to_device(qkv, mesh_device, tuple(qkv_dims))
    state_tt = _to_device(external, mesh_device, tuple(state_dims))

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return exchange_convolution_carry(qkv_tt, state_tt, sequence_parallel_axis=sp_axis)

    (entry_tt, final_tt), mismatch_markers = collect_mesh_accuracy_and_determinism_results(run)
    actual_entries = _sp_carries(entry_tt, mesh_device, sp_axis, tensor_parallel_axis)
    actual_finals = _sp_carries(final_tt, mesh_device, sp_axis, tensor_parallel_axis)

    expected_entries = [external]
    for sp_rank in range(1, sp_size):
        predecessor_end = sp_rank * local_sequence
        expected_entries.append(qkv[:, predecessor_end - history : predecessor_end])
    expected_entries_tensor = torch.stack(expected_entries)
    expected_final = qkv[:, -history:]

    assert_equal(expected_entries_tensor, actual_entries, name="halo entries")
    for sp_rank in range(sp_size):
        assert_equal(expected_final, actual_finals[sp_rank], name=f"halo final rank {sp_rank}")
    assert all(marker.item() == 0 for marker in mismatch_markers), "halo output is not bit-identical across runs"
    print(
        f"tp_axis={tensor_parallel_axis}: rank0 external carry, {sp_size - 1} neighbor carries, "
        "and all replicated final carries are exact"
    )
