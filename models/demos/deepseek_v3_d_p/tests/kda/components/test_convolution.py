# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Blackhole correctness tests for the 2D-mesh KDA convolution halo."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.chronology_oracle import chronological_topology
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections
from models.demos.deepseek_v3_d_p.tt.kda.convolution import exchange_convolution_carry
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole()]


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


@pytest.mark.parametrize(
    "mesh_device,tp_axis,device_params",
    [
        pytest.param((2, 4), 1, fabric_1d_device_params(model_config=KimiK3Config), id="SP2xTP4"),
        pytest.param((4, 2), 1, fabric_1d_device_params(model_config=KimiK3Config), id="SP4xTP2"),
        pytest.param((4, 2), 0, fabric_1d_device_params(model_config=KimiK3Config), id="SP2xTP4-axis1"),
        pytest.param((2, 4), 0, fabric_1d_device_params(model_config=KimiK3Config), id="SP4xTP2-axis1"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("local_rows", [640, 2560])
def test_exchange_convolution_carry_preserves_causal_carries(
    mesh_device: ttnn.MeshDevice, tp_axis: int, device_params: dict, local_rows: int
) -> None:
    """All sources, both axes, exact routing, trace replay and fresh-address cache hits."""
    axis = 1 - tp_axis
    sp = tuple(mesh_device.shape)[axis]
    width = 96 * 128 * 3
    generator = torch.Generator().manual_seed(128)
    qkv = torch.randn(1, sp * local_rows, width, generator=generator).bfloat16()
    dims = [None, None]
    dims[axis], dims[tp_axis] = 1, 2
    qkv_tt = _to_device(qkv, mesh_device, tuple(dims))

    def release(outputs: tuple[ttnn.Tensor, ttnn.Tensor]) -> None:
        for tensor in outputs:
            ttnn.deallocate(tensor)

    try:
        for first_rank in range(sp):
            for tail_rows in (0, 32, local_rows // 2, local_rows - 32):
                topology = chronological_topology(first_rank * local_rows + tail_rows, sp, local_rows)
                expected_entries = []
                for rank in range(sp):
                    previous = (rank - 1) % sp
                    end_row = previous * local_rows + (
                        topology.head_rows if topology.is_split and previous == first_rank else local_rows
                    )
                    outgoing = qkv[:, end_row - 3 : end_row]
                    expected_entries.append(outgoing)
                expected_entries = torch.stack(expected_entries)
                final_rank = first_rank if topology.is_split else (first_rank - 1) % sp
                end_row = (final_rank + 1) * local_rows
                expected_final = qkv[:, end_row - 3 : end_row]

                actual_start = make_actual_start(mesh_device, first_rank * local_rows + tail_rows)
                selection_records = ttnn.experimental.kda.chronological_selections(
                    actual_start, axis, local_rows, 1, 32, 32
                )
                selections = ChronologicalSelections(selection_records)

                def run() -> tuple[ttnn.Tensor, ttnn.Tensor]:
                    return exchange_convolution_carry(qkv_tt, sequence_parallel_axis=axis, selections=selections)

                def check(outputs: tuple[ttnn.Tensor, ttnn.Tensor]) -> None:
                    entries, final = outputs
                    assert entries.dtype == final.dtype == ttnn.bfloat16
                    assert entries.layout == final.layout == ttnn.ROW_MAJOR_LAYOUT
                    assert entries.memory_config() == final.memory_config() == ttnn.DRAM_MEMORY_CONFIG
                    assert torch.equal(_sp_carries(entries, mesh_device, axis, tp_axis), expected_entries)
                    assert all(
                        torch.equal(item, expected_final) for item in _sp_carries(final, mesh_device, axis, tp_axis)
                    )
                    assert final.buffer_address() != qkv_tt.buffer_address()

                for _ in range(2):
                    outputs = run()
                    check(outputs)
                    release(outputs)
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                outputs = run()
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                try:
                    for _ in range(2):
                        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                        check(outputs)
                finally:
                    ttnn.release_trace(mesh_device, trace)
                    release(outputs)

                if first_rank == sp - 1 and tail_rows in (0, local_rows // 2):
                    # Keep old allocations alive so fresh addresses cannot be recycled.
                    old_qkv = qkv_tt
                    cache_entries = mesh_device.num_program_cache_entries()
                    qkv_tt = _to_device(-qkv, mesh_device, tuple(dims))
                    try:
                        assert qkv_tt.buffer_address() != old_qkv.buffer_address()
                        expected_entries, expected_final = -expected_entries, -expected_final
                        outputs = run()
                        check(outputs)
                        release(outputs)
                        assert mesh_device.num_program_cache_entries() == cache_entries
                        assert torch.equal(
                            _sp_carries(qkv_tt, mesh_device, axis, tp_axis), -qkv.reshape(sp, 1, local_rows, width)
                        )
                    finally:
                        ttnn.deallocate(qkv_tt)
                        qkv_tt = old_qkv
                for tensor in (actual_start, selection_records):
                    ttnn.deallocate(tensor)
        assert torch.equal(_sp_carries(qkv_tt, mesh_device, axis, tp_axis), qkv.reshape(sp, 1, local_rows, width))
    finally:
        ttnn.deallocate(qkv_tt)
    print(f"SP={sp} axis={axis} C={local_rows}: exact routing, trace, rebinding and immutable inputs PASS")
