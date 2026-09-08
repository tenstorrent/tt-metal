# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All semantic chronology selectors follow changing starts through one capture."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.chronology_oracle import chronological_topology
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        ((2, 4), fabric_1d_device_params(model_config=KimiK3Config)),
        ((1, 8), fabric_1d_device_params(model_config=KimiK3Config)),
    ],
    indirect=True,
)
@pytest.mark.parametrize("sp_axis", [0, 1])
def test_chronological_selections(mesh_device, device_params, sp_axis):
    shape = tuple(mesh_device.shape)
    partitions, rows = shape[sp_axis], 640
    actual_start = make_actual_start(mesh_device)

    def device(host, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            host.bfloat16(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    qkv_host = (torch.arange(rows * 32).reshape(1, rows, 32) % 127).bfloat16()
    history_host = torch.arange(3 * partitions).reshape(1, -1, 1).expand(1, -1, 32).bfloat16()
    qkv = device(qkv_host, ttnn.ROW_MAJOR_LAYOUT)
    histories = device(history_host, ttnn.ROW_MAJOR_LAYOUT)
    transforms = device(torch.arange(1, partitions + 1).reshape(-1, 1, 1, 1).expand(-1, 1, 32, 64))
    entries = device(torch.arange(11, 11 + partitions).reshape(-1, 1, 1, 1).expand(-1, 1, 32, 32))
    finals = device(torch.arange(21, 21 + partitions).reshape(-1, 1, 1).expand(-1, 32, 32))
    prefix = device(torch.full((1, 32, 32), 99))

    def run():
        selections = ChronologicalSelections(
            ttnn.experimental.kda.chronological_selections(actual_start, sp_axis, rows, 1, 32, 32)
        )
        return (
            selections.select_outgoing_history(qkv),
            selections.select_predecessor_history(histories),
            selections.select_final_history(histories),
            selections.select_local_entry_state(entries),
            selections.select_final_state(finals, prefix),
            *(selections.select_affine_transform(transforms, step) for step in range(partitions)),
        )

    for _ in range(2):
        for tensor in run():
            ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = run()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for start in [*range(0, 2 * partitions * rows + 32, 32), 2**32 - 32]:
            source = make_actual_start(mesh_device, start)
            ttnn.copy(source, actual_start)
            ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            topology = chronological_topology(start, partitions, rows)
            shards = [[ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(t)] for t in outputs]
            for index in range(mesh_device.get_num_devices()):
                rank = index // shape[1] if sp_axis == 0 else index % shape[1]
                end = topology.head_rows if topology.is_split and rank == topology.first_rank else rows
                previous = topology.predecessor_chip(rank)
                last = topology.first_rank if topology.is_split else topology.chip_order[-1]
                expected = [
                    qkv_host[:, end - 3 : end],
                    history_host[:, 3 * previous : 3 * (previous + 1)],
                    history_host[:, 3 * last : 3 * (last + 1)],
                    torch.full((1, 1, 32, 32), 11 + topology.chip_order.index(rank)),
                    torch.full((1, 1, 32, 32), 21 + topology.first_rank if topology.is_split else 99),
                    *(torch.full((1, 1, 32, 64), 1 + physical) for physical in topology.chip_order),
                ]
                for selector, (wanted, actual) in enumerate(zip(expected, shards, strict=True)):
                    assert torch.equal(
                        wanted.bfloat16(), actual[index]
                    ), f"selector={selector} start={start} rank={rank}"
    finally:
        ttnn.release_trace(mesh_device, trace)
        for tensor in (*outputs, qkv, histories, transforms, entries, finals, prefix, actual_start):
            ttnn.deallocate(tensor)
