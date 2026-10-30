# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy test for M3VLPatchEmbed (the Conv3d patch embed as one matmul) vs the Torch reference.

The processor already flattens each patch, so pixel_values is (L, 1176). There is no class token and no
learned position embedding, so the reference's embed, the Conv3d alone, is the whole patch embed. Random
weights and pixel values (conftest.py), no checkpoint.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_patch_embed_vs_ref.py
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.patch_embed import M3VLPatchEmbed

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import MAX_REL_L2, assert_pcc, grid_name, parametrize_vision_grids, to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_patch_embed_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Patch embed matches the reference at PCC >= 0.9999 (a single matmul, nearly exact), within its relL2 bound and
    with no NaN or Inf."""
    x_in = vision_activations(grid_thw)["pixel_values"]
    reference = vision_reference()
    ref_out = reference.embed(x_in)

    tt_patch_embed = M3VLPatchEmbed.from_torch(mesh_device, reference.patch_embed, dtype=ttnn.bfloat16)
    out = to_host(tt_patch_embed(to_device(x_in, mesh_device)), mesh_device, ref_out.shape)

    assert_pcc(f"patch_embed {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.9999, MAX_REL_L2["patch_embed"])
