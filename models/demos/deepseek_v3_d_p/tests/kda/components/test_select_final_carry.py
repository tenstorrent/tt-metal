# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""select_final_carry must return the separated tail's owner state, or the prefix carry, on every trace replay."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params, torus_xy_device_params
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = run_for_blackhole()

_ROWS, _HEADS, _KEY, _VALUE = 640, 24, 128, 128


def _bounds(sp_size: int, bounded: bool) -> list[tuple[int, int | None]]:
    """Bounds replayed through one trace: aligned, rotated, split, last-rank and wrapped starts, and intervals ending
    before, exactly at and past the start of the separated tail (span - 32 rows after a 32-row offset)."""
    span = sp_size * _ROWS
    if not bounded:
        return [(start, None) for start in (0, _ROWS, _ROWS + 32, span - 32, span + 32, 32)]
    split = _ROWS + 32
    return [(split, split + span), (split, split + span - _ROWS), (0, span // 2), (32, span), (32, span - 32)]


def _meshes():
    return [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        ),
        pytest.param(
            (2, 4),
            fabric_1d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="SP2xTP4",
        ),
    ]


def _expected(rank_final: torch.Tensor, prefix: torch.Tensor, sp_size: int, start: int, end: int | None):
    """Chronology by hand: the tail separates from the prefix when the interval wraps back into its first rank."""
    offset = start % _ROWS
    first_rank = (start // _ROWS) % sp_size
    separated = sp_size > 1 and offset != 0
    if separated and end is not None:
        separated = end - start > (_ROWS - offset) + (sp_size - 1) * _ROWS
    return rank_final[first_rank][:, -1] if separated else prefix


@pytest.mark.parametrize("mesh_device,device_params", _meshes(), indirect=True)
@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
@pytest.mark.parametrize("groups", [1, 4], ids=["final", "grouped"])
def test_select_final_carry_follows_bounds_across_replays(mesh_device, bounded, groups):
    torch.manual_seed(groups)
    sp_size, tp_size = tuple(mesh_device.shape)
    # Every rank holds distinct group states, so a wrong owner, group or stale copy cannot match by chance.
    rank_final = torch.randn(sp_size, _HEADS, groups, _KEY, tp_size * _VALUE)
    prefix = torch.randn(_HEADS, _KEY, tp_size * _VALUE)
    rank_host = rank_final.reshape(sp_size * _HEADS, groups, _KEY, tp_size * _VALUE)
    if groups == 1:
        rank_host = rank_host[:, 0]
    rank_final_tt = ttnn.from_torch(
        rank_host,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, rank_host.dim() - 1), mesh_shape=(sp_size, tp_size)),
    )
    prefix_tt = ttnn.from_torch(
        prefix,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=(sp_size, tp_size)),
    )
    bounds = _bounds(sp_size, bounded)
    start_tt = make_actual_start(mesh_device, bounds[0][0])
    end_tt = make_actual_start(mesh_device, bounds[0][1]) if bounded else None

    def run():
        return ttnn.experimental.kda.select_final_carry(
            rank_final_tt,
            prefix_tt,
            actual_start=start_tt,
            actual_end=end_tt,
            local_rows=_ROWS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sequence_parallel_axis=0,
        )

    # Compile, then capture once; every replay below only changes the bounds on device.
    ttnn.deallocate(run())
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = run()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for start, end in bounds:
            for target, value in ((start_tt, start), (end_tt, end)):
                if target is not None:
                    source = make_actual_start(mesh_device, value)
                    ttnn.copy(source, target)
                    ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            expected = _expected(rank_final, prefix, sp_size, start, end)
            got = ttnn.to_torch(
                output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 2), mesh_shape=(sp_size, tp_size)),
            )
            for rank in range(sp_size):
                assert torch.equal(
                    got[rank * _HEADS : (rank + 1) * _HEADS], expected
                ), f"rank {rank} carry differs for start={start}, end={end}"
    finally:
        ttnn.release_trace(mesh_device, trace)
        for tensor in (output, rank_final_tt, prefix_tt, start_tt, end_tt):
            if tensor is not None:
                ttnn.deallocate(tensor)
