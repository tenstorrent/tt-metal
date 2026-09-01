# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Blackhole correctness tests for the 2D-mesh KDA affine prefix."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.kda.utils import collect_mesh_accuracy_and_determinism_results
from models.demos.deepseek_v3_d_p.tt.kda import ops
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_RECURRENT_STATE_DTYPE, KDARecurrenceProgramConfig
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate

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
    transform_a = transform_a.to(torch.float32)
    transform_b = transform_b.to(torch.float32)
    initial_state = initial_state.to(torch.float32)
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
    dtype: ttnn.DataType = ttnn.float32,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
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
        partitions.append(torch.cat(head_shards, dim=0))
    return torch.stack(partitions, dim=0)


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
    return torch.cat(head_shards, dim=0)


@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
def test_distributed_affine_prefix_matches_serial(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    heads = 8
    local_heads = heads // tuple(mesh_device.shape)[tensor_parallel_axis]
    dim = 32
    generator = torch.Generator().manual_seed(621 + tensor_parallel_axis)
    eye = torch.eye(dim, dtype=torch.float32).reshape(1, 1, dim, dim)
    transform_a = (0.91 * eye).expand(sp_size, heads, -1, -1).clone()
    transform_a += 0.001 * torch.randn(sp_size, heads, dim, dim, generator=generator)
    transform_b = 0.01 * torch.randn(sp_size, heads, dim, dim, generator=generator)
    initial_state = 0.01 * torch.randn(heads, dim, dim, generator=generator)

    summary_dims = [None, None]
    summary_dims[tensor_parallel_axis] = 1
    state_dims = [None, None]
    state_dims[tensor_parallel_axis] = 0
    replicated_a = _to_device(transform_a, mesh_device, tuple(summary_dims))
    replicated_b = _to_device(transform_b, mesh_device, tuple(summary_dims))
    a_tt = ttnn.reshape(ttnn.mesh_partition(replicated_a, dim=0, cluster_axis=sp_axis), (local_heads, dim, dim))
    b_tt = ttnn.reshape(ttnn.mesh_partition(replicated_b, dim=0, cluster_axis=sp_axis), (local_heads, dim, dim))
    state_tt = _to_device(initial_state, mesh_device, tuple(state_dims))

    expected_entries, expected_final = _serial_prefix(transform_a, transform_b, initial_state)
    program_config = KDARecurrenceProgramConfig()
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=program_config.affine_prefix_math_fidelity,
        fp32_dest_acc_en=True,
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        entry_tt, final_tt = ops._distributed_affine_prefix(
            a_tt,
            b_tt,
            state_tt,
            sequence_parallel_axis=sp_axis,
            compute_config=compute_config,
        )
    cache_entries = mesh_device.num_program_cache_entries()
    with ttnn.manage_config("throw_exception_on_fallback", True):
        repeated_entry_tt, repeated_final_tt = ops._distributed_affine_prefix(
            a_tt,
            b_tt,
            state_tt,
            sequence_parallel_axis=sp_axis,
            compute_config=compute_config,
        )
    ttnn.synchronize_device(mesh_device)
    assert mesh_device.num_program_cache_entries() == cache_entries
    run_index = 0

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor]:
        nonlocal run_index
        run_index += 1
        if run_index == 1:
            return entry_tt, final_tt
        if run_index == 2:
            return repeated_entry_tt, repeated_final_tt
        with ttnn.manage_config("throw_exception_on_fallback", True):
            return ops._distributed_affine_prefix(
                a_tt,
                b_tt,
                state_tt,
                sequence_parallel_axis=sp_axis,
                compute_config=compute_config,
            )

    _, mismatch_markers = collect_mesh_accuracy_and_determinism_results(run)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        traced_entry_tt, traced_final_tt = ops._distributed_affine_prefix(
            a_tt,
            b_tt,
            state_tt,
            sequence_parallel_axis=sp_axis,
            compute_config=compute_config,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    # Exercise the cached trace enough times to expose cross-rank CB reuse races.
    for _ in range(100):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    actual_entries = _partitioned_to_torch(entry_tt, mesh_device, sp_axis, tensor_parallel_axis)
    actual_final = _replicated_sp_to_torch(final_tt, mesh_device, sp_axis, tensor_parallel_axis)
    traced_entries = _partitioned_to_torch(traced_entry_tt, mesh_device, sp_axis, tensor_parallel_axis)
    traced_final = _replicated_sp_to_torch(traced_final_tt, mesh_device, sp_axis, tensor_parallel_axis)
    ttnn.release_trace(mesh_device, trace_id)

    assert entry_tt.dtype == KDA_RECURRENT_STATE_DTYPE
    assert final_tt.dtype == KDA_RECURRENT_STATE_DTYPE

    assert_accurate(expected_entries, actual_entries, name=f"tp_axis={tensor_parallel_axis} production entries")
    assert_accurate(expected_final.squeeze(0), actual_final, name=f"tp_axis={tensor_parallel_axis} production final")
    assert all(marker.item() == 0 for marker in mismatch_markers), "affine prefix is not bit-identical across runs"
    assert_accurate(expected_entries, traced_entries, name=f"tp_axis={tensor_parallel_axis} traced entries")
    assert_accurate(expected_final.squeeze(0), traced_final, name=f"tp_axis={tensor_parallel_axis} traced final")
