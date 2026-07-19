# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused trace probe for the decoder's persistent BFP8 async CCL contract."""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    get_num_links,
    get_tt_ccl,
)

PROBE_ENV = "QWEN2_5_CODER_32B_MULTICHIP_BFP8_CCL_PROBE"


def _width_sharded(rows: int, width: int, grid: ttnn.CoreGrid) -> ttnn.MemoryConfig:
    assert width % grid.num_cores == 0
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // grid.num_cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _replicated(host: torch.Tensor, mesh_device, *, dtype, memory_config):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.skipif(os.getenv(PROBE_ENV) != "1", reason="manual BFP8 async-CCL probe")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(300)
def test_persistent_bfp8_all_gather_reduce_scatter_trace(mesh_device):
    """Exercise exact decoder hidden shapes and persistent buffer placement."""

    torch.manual_seed(190727)
    batch = 32
    hidden_size = 5120
    local_hidden_size = hidden_size // 4
    local_memory = _width_sharded(batch, local_hidden_size, ttnn.CoreGrid(x=10, y=2))
    full_memory = _width_sharded(batch, hidden_size, ttnn.CoreGrid(x=8, y=2))

    host_ag = torch.empty((1, 1, batch, hidden_size), dtype=torch.bfloat16)
    for rank, chunk in enumerate(host_ag.chunk(4, dim=-1)):
        chunk.fill_(rank + 1)
    ag_input = ttnn.from_torch(
        host_ag,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    ag_input = ttnn.to_memory_config(ag_input, local_memory)

    rs_input = _replicated(
        torch.randn((1, 1, batch, hidden_size), dtype=torch.bfloat16),
        mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=full_memory,
    )
    persistent_ag = _replicated(
        torch.zeros((1, 1, batch, hidden_size), dtype=torch.bfloat16),
        mesh_device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    persistent_rs = [
        _replicated(
            torch.zeros((1, 1, batch, hidden_size), dtype=torch.bfloat16),
            mesh_device,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        _replicated(
            torch.zeros((1, 1, batch, local_hidden_size), dtype=torch.bfloat16),
            mesh_device,
            dtype=ttnn.bfloat8_b,
            memory_config=local_memory,
        ),
    ]
    ccl = get_tt_ccl(mesh_device)
    num_links = get_num_links(mesh_device)

    def run_sequence():
        ag_payload = ttnn.typecast(ag_input, ttnn.bfloat8_b)
        gathered = ttnn.experimental.all_gather_async(
            ag_payload,
            persistent_output_buffer=persistent_ag,
            dim=3,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        gathered_bf16 = ttnn.typecast(gathered, ttnn.bfloat16)
        gathered_bf16 = ttnn.to_memory_config(gathered_bf16, full_memory)

        rs_payload = ttnn.sharded_to_interleaved(rs_input, ttnn.L1_MEMORY_CONFIG)
        rs_payload = ttnn.typecast(rs_payload, ttnn.bfloat8_b)
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            rs_payload,
            persistent_output_buffers=persistent_rs,
            dim=3,
            multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=num_links,
            memory_config=local_memory,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        reduced_bf16 = ttnn.typecast(reduced, ttnn.bfloat16)
        return gathered_bf16, reduced_bf16

    run_sequence()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = run_sequence()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        first = [
            [ttnn.to_torch(device_tensor).clone() for device_tensor in ttnn.get_device_tensors(output)]
            for output in traced
        ]
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        second = [
            [ttnn.to_torch(device_tensor).clone() for device_tensor in ttnn.get_device_tensors(output)]
            for output in traced
        ]
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    assert all(torch.equal(a, b) for outputs_a, outputs_b in zip(first, second) for a, b in zip(outputs_a, outputs_b))
    assert all(torch.isfinite(tensor.float()).all() for outputs in first for tensor in outputs)
    gather_orders = [[round(float(chunk.mean())) for chunk in tensor.float().chunk(4, dim=-1)] for tensor in first[0]]
    assert all(order == [1, 2, 3, 4] for order in gather_orders)
    print(
        {
            "ag_persistent": "BFP8 DRAM",
            "rs_persistent": ["BFP8 DRAM", "BFP8 L1 WIDTH_SHARDED"],
            "gather_orders": gather_orders,
            "ag_shape_per_device": [list(tensor.shape) for tensor in first[0]],
            "rs_shape_per_device": [list(tensor.shape) for tensor in first[1]],
            "trace_bitwise_deterministic": True,
        }
    )
