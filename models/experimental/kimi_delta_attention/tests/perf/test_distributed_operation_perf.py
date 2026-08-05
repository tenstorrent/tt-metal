# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact Kimi-K3 SP4xTP2 profiler harness for distributed affine prefix."""

from __future__ import annotations

import os
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole

_BATCH = 1
_GLOBAL_HEADS = 96
_DIM = 128

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64_000_000}],
        indirect=True,
    ),
]


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


@pytest.mark.parametrize("tensor_parallel_axis", [1, 0], ids=["SP2xTP4", "SP4xTP2"])
def test_kimi_k3_distributed_affine_prefix_perf(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    """Profile ten warm trace replays at exact Kimi-K3 affine-state geometry."""
    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    sp_size = mesh_shape[sequence_parallel_axis]
    tp_size = mesh_shape[tensor_parallel_axis]
    generator = torch.Generator().manual_seed(7331)
    eye = torch.eye(_DIM, dtype=torch.float32).reshape(1, 1, _DIM, _DIM)
    transform_a = (0.97 * eye).expand(sp_size, _GLOBAL_HEADS, -1, -1).clone()
    transform_a += 0.0001 * torch.randn(transform_a.shape, generator=generator)
    transform_b = 0.001 * torch.randn(sp_size, _GLOBAL_HEADS, _DIM, _DIM, generator=generator)
    initial_state = 0.001 * torch.randn(_BATCH, _GLOBAL_HEADS, _DIM, _DIM, generator=generator)
    transform_dims: list[int | None] = [None, None]
    transform_dims[sequence_parallel_axis] = 0
    transform_dims[tensor_parallel_axis] = 1
    state_dims: list[int | None] = [None, None]
    state_dims[tensor_parallel_axis] = 1
    a_tt = _to_device(transform_a, mesh_device, transform_dims)
    b_tt = _to_device(transform_b, mesh_device, transform_dims)
    state_tt = _to_device(initial_state, mesh_device, state_dims)

    warm_entry, warm_final = ttnn.transformer._kda_distributed_affine_prefix(
        a_tt, b_tt, state_tt, sequence_parallel_axis=sequence_parallel_axis
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_entry)
    ttnn.deallocate(warm_final)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    entry_state, final_state = ttnn.transformer._kda_distributed_affine_prefix(
        a_tt, b_tt, state_tt, sequence_parallel_axis=sequence_parallel_axis
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    repetitions = int(os.getenv("PERF_REPS", "10"))
    signpost(header="distributed_affine_prefix_start")
    start = time.perf_counter()
    for _ in range(repetitions):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    signpost(header="distributed_affine_prefix_stop")
    print(
        f"Distributed affine prefix SP{sp_size}xTP{tp_size} "
        f"shape=[1,{_GLOBAL_HEADS // tp_size},{_DIM},{_DIM}]: "
        f"transport=all_gather: wall={elapsed * 1e3 / repetitions:.3f} ms/replay "
        f"over {repetitions} replays"
    )

    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(entry_state)
    ttnn.deallocate(final_state)
