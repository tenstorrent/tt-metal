# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for M3VLPatchEmbed (the Conv3d patch embed as one matmul) vs the Torch reference.

The processor already flattens each patch, so pixel_values is (L, 1176). There is no class token and no
learned position embedding, so the patch embed output is the reference's pre_layrnorm input. Random
weights and pixel values (conftest.py), no checkpoint.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_patch_embed_vs_ref.py
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.patch_embed import M3VLPatchEmbed

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import assert_pcc, grid_name, parametrize_vision_grids, to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_patch_embed_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Patch embed matches the reference at PCC >= 0.9999 (a single matmul, nearly exact)."""
    acts = vision_activations(grid_thw)
    x_in, ref_out = acts["pixel_values"], acts["pre_layrnorm.in"]

    tt_patch_embed = M3VLPatchEmbed.from_torch(mesh_device, vision_reference().patch_embed, dtype=ttnn.bfloat16)
    out = to_host(tt_patch_embed(to_device(x_in, mesh_device)), mesh_device, ref_out.shape)

    assert_pcc(f"patch_embed {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.9999)
