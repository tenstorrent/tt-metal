# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fabric convolution-history gather and exchange against an independent chronology oracle, under one trace."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.chronology_oracle import chronological_topology
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = run_for_blackhole()


@pytest.mark.parametrize(
    "mesh_device,device_params,sp_axis",
    [
        pytest.param((2, 4), fabric_1d_device_params(), 0, id="SP2-fabric-1d"),
        pytest.param((2, 4), fabric_1d_device_params(), 1, id="SP4-fabric-1d"),
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            0,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8-torus-xy",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
def test_exchange_histories(mesh_device, device_params, sp_axis, bounded):
    shape = tuple(mesh_device.shape)
    partitions, rows, width = shape[sp_axis], 640, 9216
    capacity = partitions * rows
    # Distinct random rows per rank, with columns past the channels that the exchange must skip.
    torch.manual_seed(0)
    host = torch.randn(partitions, rows, width + 256).bfloat16()
    mesh_dims = [None, None]
    mesh_dims[sp_axis] = 0
    projected = ttnn.from_torch(
        host.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(mesh_dims), mesh_shape=shape),
    )
    actual_start = make_actual_start(mesh_device)
    actual_end = make_actual_start(mesh_device, capacity) if bounded else None

    def run():
        return ttnn.experimental.kda.exchange_histories(
            projected,
            width=width,
            actual_start=actual_start,
            local_rows=rows,
            actual_end=actual_end,
            sequence_parallel_axis=sp_axis,
        )

    for _ in range(2):
        for tensor in run():
            ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = run()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        lengths = [capacity, 32, rows - 32, rows, min(rows + 32, capacity), capacity - 32] if bounded else [capacity]
        starts = [0, 32, rows, rows + 96, 2 * capacity + 32]
        for start in starts:
            for length in lengths:
                for destination, value in ((actual_start, start), (actual_end, start + length)):
                    if destination is not None:
                        source = make_actual_start(mesh_device, value)
                        ttnn.copy(source, destination)
                        ttnn.deallocate(source)
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                topology = chronological_topology(start, partitions, rows)
                # Each local row's chronological index; the last valid token's row ends the final history.
                head, steps = topology.head_rows, (torch.arange(partitions) - topology.first_rank) % partitions
                local = torch.arange(rows)
                chronology = torch.stack(
                    [
                        torch.where(local < head, local, head + (partitions - 1) * rows + local - head)
                        if step == 0
                        else head + (step - 1) * rows + local
                        for step in steps.tolist()
                    ]
                ).flatten()
                last_row = int(torch.nonzero(chronology == length - 1))
                last, last_local = last_row // rows, last_row % rows
                expected_final = host[last, last_local - 2 : last_local + 1, :width].reshape(1, 3, width)
                shards = [[ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(t)] for t in outputs]
                for index in range(mesh_device.get_num_devices()):
                    rank = index // shape[1] if sp_axis == 0 else index % shape[1]
                    previous = topology.predecessor_chip(rank)
                    # The outgoing history ends the predecessor's segment, or its head segment when it is split.
                    end = topology.head_rows if topology.is_split and previous == topology.first_rank else rows
                    expected_predecessor = host[previous, end - 3 : end, :width].reshape(1, 3, width)
                    case = f"start={start} length={length} rank={rank}"
                    assert torch.equal(shards[0][index], expected_predecessor), f"predecessor {case}"
                    assert torch.equal(shards[1][index], expected_final), f"final {case}"
    finally:
        ttnn.release_trace(mesh_device, trace)
        for tensor in (*outputs, projected, actual_start, actual_end):
            if tensor is not None:
                ttnn.deallocate(tensor)
