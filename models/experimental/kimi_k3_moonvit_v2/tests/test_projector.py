# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Projector` against the reference `mm_projector`, which runs on the CPU."""

from __future__ import annotations

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    random_tensor,
    assert_quality,
    device_config,
    row_grid,
    ROW_GRIDS,
    NOT_ON_SIMULATOR,
    REFERENCE,
    P150_AND_RING,
    SIMULATOR_GRID,
    SIMULATOR_ROWS,
    SINGLE_CHIP,
    MeshRows,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.projector import TT_Projector


def check_projector(mesh_device, sp_axis, config, reference, grid, padded_patches: int) -> None:
    """The merged tokens of `grid`, padded to `padded_patches / 4` (the ring's length after merging);
    one chip runs its quarter (`MeshRows`)."""
    ckc = compute_kernel_config(mesh_device)
    tt = TT_Projector(device_config(config, sp_axis), mesh_device=mesh_device, compute_kernel_config=ckc)
    tt.load_torch_state_dict(reference.mm_projector.state_dict())

    _, h, w = grid[0]
    rows = MeshRows(h * w // config.merge_area, padded_patches // config.merge_area, sp_axis)
    x = random_tensor(rows.n, config.merge_area, config.hidden_size)
    with torch.no_grad():
        (golden,) = reference.mm_projector([rows.take(x)])

    out = tt(rows.shard(x.reshape(rows.n, config.merged_dim), mesh_device))
    assert_quality(golden, rows.gather(out, mesh_device), relative_rmse=0.01)


@NOT_ON_SIMULATOR
@P150_AND_RING
@REFERENCE
@ROW_GRIDS
def test_projector(mesh_device, sp_axis, num_links, config, reference, size):
    check_projector(mesh_device, sp_axis, config, reference, row_grid(size), config.max_patches)


@SINGLE_CHIP
@pytest.mark.parametrize("reference", ["random"], indirect=True)
def test_projector_small(mesh_device, sp_axis, num_links, config, reference):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS patches."""
    check_projector(mesh_device, sp_axis, config, reference, SIMULATOR_GRID, SIMULATOR_ROWS)
