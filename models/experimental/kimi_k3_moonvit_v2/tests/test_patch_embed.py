# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Patch_Embed` against the reference `patch_embed`."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    DEVICE_GRIDS,
    device_config,
    NOT_WRITTEN,
    REFERENCE,
    P150_AND_RING,
    random_patches,
    sp_gather,
    sp_shard,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import check_grid, compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.reference import functional
from models.experimental.kimi_k3_moonvit_v2.tt.patch_embed import TT_Patch_Embed


@P150_AND_RING
@REFERENCE
@DEVICE_GRIDS
@NOT_WRITTEN
def test_patch_embed(mesh_device, sp_axis, num_links, config, reference, grid):
    config = device_config(config, sp_axis)
    tt = TT_Patch_Embed(config, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config(mesh_device))
    tt.load_torch_state_dict(reference.vision_tower.patch_embed.state_dict())

    pixels, grid_thws = random_patches(config, grid)
    h, w = check_grid(grid_thws, config)
    with torch.no_grad():
        golden = reference.vision_tower.patch_embed(pixels.to(torch.bfloat16), grid_thws)

    pos_emb = functional.pos_embeds(tt.pos_emb_weight, h, w, config.pos_emb_interpolation_mode)
    out = tt(
        sp_shard(functional.patches_to_rows(pixels), mesh_device, sp_axis, config.max_patches),
        sp_shard(pos_emb, mesh_device, sp_axis, config.max_patches),
    )
    assert_pcc(golden, sp_gather(out, mesh_device, sp_axis, h * w))
