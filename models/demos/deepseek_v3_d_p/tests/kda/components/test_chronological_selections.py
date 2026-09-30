# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All semantic chronology selectors follow changing bounds through one capture."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.chronology_oracle import chronological_topology
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [((2, 4), fabric_1d_device_params()), ((1, 8), fabric_1d_device_params())],
    indirect=True,
)
@pytest.mark.parametrize("sp_axis", [0, 1])
@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
def test_chronological_selections(mesh_device, device_params, sp_axis, bounded):
    shape = tuple(mesh_device.shape)
    partitions, rows = shape[sp_axis], 640
    actual_start = make_actual_start(mesh_device)
    capacity = partitions * rows
    actual_end = make_actual_start(mesh_device, capacity) if bounded else None

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
            ttnn.experimental.kda.chronological_selections(
                actual_start, sp_axis, rows, 1, 32, 32, actual_end=actual_end
            )
        )
        return (
            selections.select_outgoing_history(qkv),
            selections.select_predecessor_history(histories),
            selections.select_final_history(histories),
            selections.select_local_entry_state(entries),
            selections.select_final_state(finals, prefix),
            selections.select_local_final_history(qkv, partitions),
            *(selections.select_affine_transform(transforms, step) for step in range(partitions)),
        )

    for _ in range(2):
        for tensor in run():
            ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = run()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        if bounded:
            # End before/at/after device boundaries and in a separated tail, then
            # return to full capacity, updating both scalars without recapture.
            lengths = [
                capacity,
                32,
                96,
                rows - 32,
                rows,
                min(rows + 32, capacity),
                capacity - 64,
                capacity - 32,
                capacity,
            ]
            cases = [(start, length) for start in (0, 32, 96, rows, rows + 96, 2 * capacity + 32) for length in lengths]
        else:
            cases = [(start, capacity) for start in [*range(0, 2 * capacity + 32, 32), 2**32 - 32]]
        for start, length in cases:
            for destination, value in ((actual_start, start), (actual_end, start + length)):
                if destination is not None:
                    source = make_actual_start(mesh_device, value)
                    ttnn.copy(source, destination)
                    ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            topology = chronological_topology(start, partitions, rows)
            # Enumerate absolute token ownership independently of the native
            # interval formulas. Local valid tokens occupy a packed prefix.
            owners = ((torch.arange(length, dtype=torch.int64) + start) // rows) % partitions
            valid_rows = torch.bincount(owners, minlength=partitions).tolist()
            last = int(owners[-1])
            has_tail = last == int(owners[0]) and bool(torch.any(owners != owners[0]))
            shards = [[ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(t)] for t in outputs]
            for index in range(mesh_device.get_num_devices()):
                rank = index // shape[1] if sp_axis == 0 else index % shape[1]
                end = topology.head_rows if topology.is_split and rank == topology.first_rank else rows
                previous = topology.predecessor_chip(rank)
                local_history_end = valid_rows[rank] or 3
                expected = [
                    qkv_host[:, end - 3 : end],
                    history_host[:, 3 * previous : 3 * (previous + 1)],
                    history_host[:, 3 * last : 3 * (last + 1)],
                    torch.full((1, 1, 32, 32), 11 + topology.chip_order.index(rank)),
                    torch.full((1, 1, 32, 32), 21 + last if has_tail else 99),
                    qkv_host[:, local_history_end - 3 : local_history_end],
                    *(torch.full((1, 1, 32, 64), 1 + physical) for physical in topology.chip_order),
                ]
                for selector, (wanted, actual) in enumerate(zip(expected, shards, strict=True)):
                    assert torch.equal(
                        wanted.bfloat16(), actual[index]
                    ), f"selector={selector} start={start} length={length} rank={rank}"
    finally:
        ttnn.release_trace(mesh_device, trace)
        for tensor in (*outputs, qkv, histories, transforms, entries, finals, prefix, actual_start, actual_end):
            if tensor is not None:
                ttnn.deallocate(tensor)
