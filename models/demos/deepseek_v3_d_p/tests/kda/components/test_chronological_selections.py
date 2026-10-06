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
    layer_history_host = torch.arange(500, 503).reshape(1, 3, 1).expand(1, 3, 32).bfloat16()
    layer_history = device(layer_history_host, ttnn.ROW_MAJOR_LAYOUT)
    finals = device(torch.arange(21, 21 + partitions).reshape(-1, 1, 1).expand(-1, 32, 32))
    prefix = device(torch.full((1, 32, 32), 99))

    def run():
        selections = ChronologicalSelections(
            ttnn.experimental.kda.chronological_selections(
                actual_start, sp_axis, rows, 1, 32, 32, actual_end=actual_end
            )
        )
        predecessor = selections.select_predecessor_history(histories)
        return (
            selections.select_outgoing_history(qkv),
            predecessor,
            selections.select_final_history(histories),
            selections.select_final_state(finals, prefix),
            selections.select_local_final_history(qkv, layer_history, predecessor),
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
                # Unaligned ends, including end segments with one or two valid rows
                # that continue the layer history, a predecessor, or a separated tail.
                35,
                min(rows + 35, capacity),
                capacity - 1,
                1,
                2,
                min(rows + 1, capacity),
                min(rows + 2, capacity),
                capacity - 31,
                capacity - 94,
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
                # The end segment is the separated tail when it holds valid rows; its
                # history continues the tokens before that segment.
                tail = has_tail and rank == last
                segment_begin = topology.head_rows if tail else 0
                preceding = (
                    layer_history_host
                    if rank == topology.first_rank and not tail
                    else history_host[:, 3 * previous : 3 * (previous + 1)]
                )
                local_history = torch.cat([preceding, qkv_host[:, segment_begin : valid_rows[rank]]], dim=1)[:, -3:]
                expected = [
                    qkv_host[:, end - 3 : end],
                    history_host[:, 3 * previous : 3 * (previous + 1)],
                    history_host[:, 3 * last : 3 * (last + 1)],
                    torch.full((1, 1, 32, 32), 21 + last if has_tail else 99),
                    local_history,
                ]
                for selector, (wanted, actual) in enumerate(zip(expected, shards, strict=True)):
                    if selector == 4 and valid_rows[rank] == 0:
                        continue  # An empty rank's local history is unspecified and ignored.
                    assert torch.equal(
                        wanted.bfloat16(), actual[index]
                    ), f"selector={selector} start={start} length={length} rank={rank}"
    finally:
        ttnn.release_trace(mesh_device, trace)
        for tensor in (*outputs, qkv, histories, layer_history, finals, prefix, actual_start, actual_end):
            if tensor is not None:
                ttnn.deallocate(tensor)
