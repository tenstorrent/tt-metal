# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blackhole correctness tests for the 2D-mesh KDA affine prefix."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 4_000_000}],
        indirect=True,
    ),
]


AffineTransform = tuple[torch.Tensor, torch.Tensor]


def _compose(
    after_a: torch.Tensor,
    after_b: torch.Tensor,
    before_a: torch.Tensor,
    before_b: torch.Tensor,
) -> AffineTransform:
    return after_a @ before_a, after_a @ before_b + after_b


def _serial_prefix(
    transform_a: torch.Tensor,
    transform_b: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    entries = [initial_state]
    prefix_a, prefix_b = transform_a[0], transform_b[0]
    for rank in range(1, transform_a.shape[0]):
        entries.append(prefix_a @ initial_state + prefix_b)
        prefix_a, prefix_b = _compose(transform_a[rank], transform_b[rank], prefix_a, prefix_b)
    return torch.stack(entries), (prefix_a @ initial_state + prefix_b).unsqueeze(0)


def _to_device(
    tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    mesh_dims: tuple[int | None, int | None],
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=mesh_dims, mesh_shape=tuple(mesh_device.shape)),
    )


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _coordinate(sp_rank: int, tp_rank: int, sp_axis: int) -> tuple[int, int]:
    return (sp_rank, tp_rank) if sp_axis == 0 else (tp_rank, sp_rank)


def _partitioned_to_torch(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
) -> torch.Tensor:
    shards = _host_shards(tensor)
    rows, columns = tuple(mesh_device.shape)
    sp_size = (rows, columns)[sp_axis]
    tp_size = (rows, columns)[tp_axis]
    partitions = []
    for sp_rank in range(sp_size):
        head_shards = []
        for tp_rank in range(tp_size):
            row, column = _coordinate(sp_rank, tp_rank, sp_axis)
            head_shards.append(shards[row * columns + column])
        partitions.append(torch.cat(head_shards, dim=1))
    return torch.cat(partitions, dim=0)


def _replicated_sp_to_torch(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
) -> torch.Tensor:
    shards = _host_shards(tensor)
    _, columns = tuple(mesh_device.shape)
    tp_size = tuple(mesh_device.shape)[tp_axis]
    head_shards = []
    for tp_rank in range(tp_size):
        row, column = _coordinate(0, tp_rank, sp_axis)
        head_shards.append(shards[row * columns + column])
    return torch.cat(head_shards, dim=1)


def _all_gather_oracle(
    transform_a: ttnn.Tensor,
    transform_b: ttnn.Tensor,
    initial_state: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gathered_a = ttnn.all_gather(transform_a, dim=0, cluster_axis=sp_axis, topology=ttnn.Topology.Linear)
    gathered_b = ttnn.all_gather(transform_b, dim=0, cluster_axis=sp_axis, topology=ttnn.Topology.Linear)
    host_a = _replicated_sp_to_torch(gathered_a, mesh_device, sp_axis, tp_axis)
    host_b = _replicated_sp_to_torch(gathered_b, mesh_device, sp_axis, tp_axis)
    return _serial_prefix(host_a, host_b, initial_state)


def _assert_close(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    passed, pcc = comp_pcc(expected, actual, pcc=0.999)
    max_abs = (expected - actual).abs().max().item()
    print(f"{name}: PCC={pcc:.6f}, max_abs={max_abs:.6e}")
    assert passed, f"{name} PCC {pcc:.6f} < 0.999"


@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
def test_distributed_affine_prefix_matches_serial_and_all_gather(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    heads = 8
    dim = 32
    generator = torch.Generator().manual_seed(621 + tensor_parallel_axis)
    eye = torch.eye(dim, dtype=torch.float32).reshape(1, 1, dim, dim)
    transform_a = (0.91 * eye).expand(sp_size, heads, -1, -1).clone()
    transform_a += 0.001 * torch.randn(sp_size, heads, dim, dim, generator=generator)
    transform_b = 0.01 * torch.randn(sp_size, heads, dim, dim, generator=generator)
    initial_state = 0.01 * torch.randn(heads, dim, dim, generator=generator)
    identity_a = eye.expand(1, heads, -1, -1).clone()
    zero_b = torch.zeros_like(identity_a)

    summary_dims = [None, None]
    summary_dims[sp_axis] = 0
    summary_dims[tensor_parallel_axis] = 1
    state_dims = [None, None]
    state_dims[tensor_parallel_axis] = 1
    a_tt = _to_device(transform_a, mesh_device, tuple(summary_dims))
    b_tt = _to_device(transform_b, mesh_device, tuple(summary_dims))
    state_tt = _to_device(initial_state.unsqueeze(0), mesh_device, tuple(state_dims))
    identity_tt = _to_device(identity_a, mesh_device, tuple(state_dims))
    zero_tt = _to_device(zero_b, mesh_device, tuple(state_dims))

    expected_entries, expected_final = _serial_prefix(transform_a, transform_b, initial_state)
    oracle_entries, oracle_final = _all_gather_oracle(
        a_tt, b_tt, initial_state, mesh_device, sp_axis, tensor_parallel_axis
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        entry_tt, final_tt = ttnn.transformer.kda_distributed_affine_prefix(
            a_tt,
            b_tt,
            state_tt,
            identity_tt,
            zero_tt,
            sequence_parallel_axis=sp_axis,
        )
    cache_entries = mesh_device.num_program_cache_entries()
    with ttnn.manage_config("throw_exception_on_fallback", True):
        repeated_entry_tt, repeated_final_tt = ttnn.transformer.kda_distributed_affine_prefix(
            a_tt,
            b_tt,
            state_tt,
            identity_tt,
            zero_tt,
            sequence_parallel_axis=sp_axis,
        )
    ttnn.synchronize_device(mesh_device)
    assert mesh_device.num_program_cache_entries() == cache_entries

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        traced_entry_tt, traced_final_tt = ttnn.transformer.kda_distributed_affine_prefix(
            a_tt,
            b_tt,
            state_tt,
            identity_tt,
            zero_tt,
            sequence_parallel_axis=sp_axis,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    actual_entries = _partitioned_to_torch(entry_tt, mesh_device, sp_axis, tensor_parallel_axis)
    actual_final = _replicated_sp_to_torch(final_tt, mesh_device, sp_axis, tensor_parallel_axis)
    repeated_entries = _partitioned_to_torch(repeated_entry_tt, mesh_device, sp_axis, tensor_parallel_axis)
    repeated_final = _replicated_sp_to_torch(repeated_final_tt, mesh_device, sp_axis, tensor_parallel_axis)
    traced_entries = _partitioned_to_torch(traced_entry_tt, mesh_device, sp_axis, tensor_parallel_axis)
    traced_final = _replicated_sp_to_torch(traced_final_tt, mesh_device, sp_axis, tensor_parallel_axis)
    ttnn.release_trace(mesh_device, trace_id)

    _assert_close(f"tp_axis={tensor_parallel_axis} oracle entries", expected_entries, oracle_entries)
    _assert_close(f"tp_axis={tensor_parallel_axis} oracle final", expected_final, oracle_final)
    _assert_close(f"tp_axis={tensor_parallel_axis} production entries", expected_entries, actual_entries)
    _assert_close(f"tp_axis={tensor_parallel_axis} production final", expected_final, actual_final)
    _assert_close(f"tp_axis={tensor_parallel_axis} repeated entries", expected_entries, repeated_entries)
    _assert_close(f"tp_axis={tensor_parallel_axis} repeated final", expected_final, repeated_final)
    _assert_close(f"tp_axis={tensor_parallel_axis} traced entries", expected_entries, traced_entries)
    _assert_close(f"tp_axis={tensor_parallel_axis} traced final", expected_final, traced_final)
