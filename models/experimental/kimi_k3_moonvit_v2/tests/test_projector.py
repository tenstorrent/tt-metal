# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Projector` against the reference `mm_projector`, which runs on the CPU."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    SYNTHETIC,
    REAL,
    random_tensor,
    assert_quality,
    device_config,
    padding_layout_grid,
    PADDING_LAYOUT_GRIDS,
    P150,
    SIMULATOR_GRID,
    SIMULATOR,
    MeshRows,
    simulator_config,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.projector import TT_Projector


def check_projector(mesh_device, sp_axis, config, reference, grid) -> None:
    """The merged tokens of `grid`, padded to `config.max_patches / 4` (the ring's length after merging);
    one chip runs its quarter (`MeshRows`)."""
    ckc = compute_kernel_config(mesh_device)
    tt = TT_Projector(device_config(config, sp_axis), mesh_device=mesh_device, compute_kernel_config=ckc)
    tt.load_torch_state_dict(reference.mm_projector.state_dict())

    _, h, w = grid[0]
    rows = MeshRows(h * w // config.merge_area, config.max_patches // config.merge_area, sp_axis)
    x = random_tensor(rows.n, config.merge_area, config.hidden_size)
    with torch.no_grad():
        (golden,) = reference.mm_projector([rows.real_rows(x)])

    out = tt(rows.shard(x.reshape(rows.n, config.merged_dim), mesh_device))
    assert_quality(golden, rows.gather(out, mesh_device), relative_rmse=0.01)


@P150
@SYNTHETIC
@REAL
@PADDING_LAYOUT_GRIDS
def test_projector(mesh_device, sp_axis, num_links, config, reference, size):
    check_projector(mesh_device, sp_axis, config, reference, padding_layout_grid(size))


@SIMULATOR
@SYNTHETIC
def test_projector_small(mesh_device, sp_axis, num_links, config, reference):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS patches."""
    check_projector(mesh_device, sp_axis, simulator_config(config), reference, SIMULATOR_GRID)
