# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Patch_Embed` then `TT_MoonViT_Encoder` (ring attention over the 4-chip ring)
against the reference `vision_tower.patch_embed` then `vision_tower.encoder`, first TEST_LAYERS
layers.

Covers the encoder without the patch merger and projector. The inputs come from
`host_prep.prepare_host_inputs`, as in the full model: window order, padded to `max_patches`. The
output is put back in raster order to compare."""

from __future__ import annotations

import torch

import ttnn

from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    ATTENTION_GRIDS,
    RING,
    SHALLOW_REFERENCE,
    random_patches,
    ring_parallel,
    sp_gather,
    sp_shard,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import check_grid, compute_kernel_config, resolve_parallel
from models.experimental.kimi_k3_moonvit_v2.tt.patch_embed import TT_Patch_Embed
from models.experimental.kimi_k3_moonvit_v2.tt.rope import TT_Rope
from models.experimental.kimi_k3_moonvit_v2.tt.vision_attention import logical_length_tensor
from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Encoder


@RING
@SHALLOW_REFERENCE
@ATTENTION_GRIDS
def test_vision_encoder(mesh_device, sp_axis, num_links, shallow_config, shallow_reference, grid):
    config = shallow_config
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    ckc = compute_kernel_config(mesh_device)
    tower = shallow_reference.vision_tower
    patch_embed = TT_Patch_Embed(config, mesh_device=mesh_device, compute_kernel_config=ckc)
    patch_embed.load_torch_state_dict(tower.patch_embed.state_dict())
    encoder = TT_MoonViT_Encoder(config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=ckc)
    encoder.load_torch_state_dict(tower.encoder.state_dict())

    pixels, grid_thws = random_patches(config, grid)
    h, w = check_grid(grid_thws, config)
    with torch.no_grad():
        golden = tower.encoder(tower.patch_embed(pixels.to(torch.bfloat16), grid_thws), grid_thws)

    host = host_prep.prepare_host_inputs(config, patch_embed.pos_emb_weight, pixels, grid_thws)
    rope = TT_Rope(config, mesh_device=mesh_device, parallel=parallel).write(host["cos"], host["sin"])
    patches = sp_shard(host["patches"], mesh_device, sp_axis, config.max_patches)
    pos_emb = sp_shard(host["pos_emb"], mesh_device, sp_axis, config.max_patches, ttnn.float32)
    logical_n = logical_length_tensor(mesh_device, host["logical_n"])

    x = patch_embed(patches, pos_emb)
    out = encoder(x, rope, logical_n)
    out = sp_gather(out, mesh_device, sp_axis, h * w)
    assert_pcc(golden, host_prep.from_window_order(out, h, w, config.merge_kernel_size))
