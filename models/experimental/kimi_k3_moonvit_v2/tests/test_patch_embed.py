# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Patch_Embed` against the reference `patch_embed`, on the window-ordered, padded rows
`host_prep.prepare_host_inputs` builds for the model."""

from __future__ import annotations

import torch

import ttnn

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_quality,
    device_config,
    row_grid,
    ROW_GRIDS,
    REFERENCE,
    MeshRows,
    P150_AND_RING,
    SIMULATOR_GRID,
    SIMULATOR,
    random_patches,
    simulator_config,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import check_grid, compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tt.patch_embed import TT_Patch_Embed


def check_patch_embed(mesh_device, sp_axis, config, reference, grid) -> None:
    """`grid` padded to `config.max_patches`, the ring's length; one chip runs its quarter (`MeshRows`)."""
    ckc = compute_kernel_config(mesh_device)
    tt = TT_Patch_Embed(device_config(config, sp_axis), mesh_device=mesh_device, compute_kernel_config=ckc)
    tt.load_torch_state_dict(reference.vision_tower.patch_embed.state_dict())

    pixels, grid_thws = random_patches(config, grid)
    h, w = check_grid(grid_thws, config)
    with torch.no_grad():
        golden = reference.vision_tower.patch_embed(pixels.to(torch.bfloat16), grid_thws)
    golden = host_prep.to_window_order(golden, h, w, config.merge_kernel_size)

    host = host_prep.prepare_host_inputs(config, tt.pos_emb_weight, pixels, grid_thws)
    rows = MeshRows(h * w, config.max_patches, sp_axis)
    out = tt(rows.shard(host["patches"], mesh_device), rows.shard(host["pos_emb"], mesh_device, ttnn.float32))
    assert_quality(rows.take(golden), rows.gather(out, mesh_device), relative_rmse=0.005)


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
def test_patch_embed(mesh_device, sp_axis, num_links, config, reference, size):
    check_patch_embed(mesh_device, sp_axis, config, reference, row_grid(size))


@SIMULATOR
def test_patch_embed_small(mesh_device, sp_axis, num_links, config, reference):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS."""
    check_patch_embed(mesh_device, sp_axis, simulator_config(config), reference, SIMULATOR_GRID)
