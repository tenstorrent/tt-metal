# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: the full `TT_MoonViT_Vision_Model` against the reference `KimiK3VisionModel` (first TEST_LAYERS layers)."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    DEVICE_GRIDS,
    NOT_WRITTEN,
    SHALLOW_REFERENCE,
    RING,
    random_patches,
    ring_parallel,
)
from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Vision_Model


def build(mesh_device, sp_axis, num_links, config, shallow_reference) -> TT_MoonViT_Vision_Model:
    sequence_parallel, ccl = ring_parallel(mesh_device, sp_axis, num_links)
    tt = TT_MoonViT_Vision_Model(config, mesh_device=mesh_device, sequence_parallel=sequence_parallel, ccl=ccl)
    tt.load_torch_state_dict(shallow_reference.state_dict())
    return tt


@RING
@SHALLOW_REFERENCE
def test_load_weights(mesh_device, sp_axis, num_links, config, shallow_reference):
    tt = build(mesh_device, sp_axis, num_links, config, shallow_reference)
    assert tt.is_loaded()
    assert torch.equal(
        tt.vision_tower.patch_embed.pos_emb_weight, shallow_reference.vision_tower.patch_embed.pos_emb.weight
    )


@RING
@SHALLOW_REFERENCE
@DEVICE_GRIDS
@NOT_WRITTEN
def test_vision_model(mesh_device, sp_axis, num_links, config, shallow_reference, grid):
    tt = build(mesh_device, sp_axis, num_links, config, shallow_reference)
    pixels, grid_thws = random_patches(config, grid)
    with torch.no_grad():
        (golden,) = shallow_reference(pixels.to(torch.bfloat16), grid_thws)
    assert_pcc(golden, tt.run(pixels, grid_thws))
