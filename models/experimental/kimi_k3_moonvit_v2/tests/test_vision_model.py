# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: the full `TT_MoonViT_Vision_Model` (patch embed, all 27 encoder blocks, final norm, patch merger,
projector) against the reference `KimiK3VisionModel` in fp32 (`goldens.to_fp32`), on small images.
Not the bf16 reference: its own rounding is larger than the device's error (on 64x64 with the
checkpoint, PCC 0.848 against fp32). `test_full_size.py` runs it on a full-size image against the
saved fp32 outputs."""

from __future__ import annotations

import copy

import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_quality,
    ATTENTION_GRIDS,
    REFERENCE,
    RING,
    random_patches,
    ring_parallel,
)
from models.experimental.kimi_k3_moonvit_v2.tests.goldens import to_fp32
from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Vision_Model


def build(mesh_device, sp_axis, num_links, config, reference) -> TT_MoonViT_Vision_Model:
    sequence_parallel, ccl = ring_parallel(mesh_device, sp_axis, num_links)
    tt = TT_MoonViT_Vision_Model(config, mesh_device=mesh_device, sequence_parallel=sequence_parallel, ccl=ccl)
    tt.load_torch_state_dict(reference.state_dict())
    return tt


@RING
@REFERENCE
def test_load_weights(mesh_device, sp_axis, num_links, config, reference):
    tt = build(mesh_device, sp_axis, num_links, config, reference)
    assert tt.is_loaded()
    assert torch.equal(tt.vision_tower.patch_embed.pos_emb_weight, reference.vision_tower.patch_embed.pos_emb.weight)


@RING
@REFERENCE
@ATTENTION_GRIDS
def test_vision_model(mesh_device, sp_axis, num_links, config, reference, grid):
    tt = build(mesh_device, sp_axis, num_links, config, reference)
    pixels, grid_thws = random_patches(config, grid)
    with torch.no_grad():
        (golden,) = to_fp32(copy.deepcopy(reference))(pixels.float(), grid_thws)
    assert_quality(golden, tt.run(pixels, grid_thws), relative_rmse=0.07)
