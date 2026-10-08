# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fabric final-state selection must match the gather-and-select reference bit for bit."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import select_final_state
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

_ROWS, _HEADS, _KEY, _VALUE = 640, 24, 128, 128
_REPEATS = 3


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "start,end",
    [(0, None), (640, None), (672, None), (3200, None), (672, 672 + 5120), (672, 672 + 2048), (0, 1280)],
    ids=["aligned", "rotated", "split", "late-split", "split-full-end", "split-short-end", "aligned-short-end"],
)
@pytest.mark.parametrize("groups", [1, 4], ids=["final", "grouped"])
def test_select_final_carry_matches_reference(mesh_device, start, end, groups):
    torch.manual_seed(start + (end or 0))
    sp_size, tp_size = tuple(mesh_device.shape)
    # Every rank holds a distinct local final, so a wrong owner or a stale copy cannot match by chance.
    # Grouped inputs hold every group's state; only the last group is the rank's final.
    rank_shape = (
        (sp_size * _HEADS, _KEY, tp_size * _VALUE)
        if groups == 1
        else (sp_size * _HEADS, groups, _KEY, tp_size * _VALUE)
    )
    rank_final = ttnn.from_torch(
        torch.randn(rank_shape),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, len(rank_shape) - 1), mesh_shape=(sp_size, tp_size)),
    )
    prefix = ttnn.from_torch(
        torch.randn(_HEADS, _KEY, tp_size * _VALUE),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=(sp_size, tp_size)),
    )
    actual_start = make_actual_start(mesh_device, start)
    actual_end = None if end is None else make_actual_start(mesh_device, end)
    selections = ChronologicalSelections(
        ttnn.experimental.kda.chronological_selections(
            actual_start, 0, _ROWS, _HEADS, _KEY, _VALUE, actual_end=actual_end
        )
    )

    def select(fused):
        state = select_final_state(
            rank_final,
            prefix,
            selections=selections,
            actual_start=actual_start,
            actual_end=actual_end,
            local_rows=_ROWS,
            sequence_parallel_axis=0,
            fused=fused,
        )
        state = ttnn.reshape(state, (_HEADS, _KEY, _VALUE))
        return ttnn.to_torch(state, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    reference = select(False)
    # Repeats replay the cached program, so they also check that the line semaphores reset.
    for repeat in range(_REPEATS):
        assert torch.equal(select(True), reference), f"fused final state differs from the reference on call {repeat}"
