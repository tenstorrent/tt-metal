# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Device test of M3VLPatchEmbed against Hugging Face's MiniMaxM3VLVisionEmbeddings.

M3VLPatchEmbed runs the Conv3d patch embedding as one matmul. The weights and pixel values are seeded random values
(tests/conftest.py), so no checkpoint is needed.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.patch_embed import M3VLPatchEmbed

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import MAX_REL_L2, assert_close, grid_name, parametrize_vision_grids, rows_to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_patch_embed_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Patch embed matches the reference at PCC >= 0.9999 (a single matmul, nearly exact), within its relL2 bound and
    with no NaN or Inf."""
    x_in = vision_activations(grid_thw)["pixel_values"]
    embeddings = vision_reference().tower.embeddings
    ref_out = embeddings(x_in)

    tt_patch_embed = M3VLPatchEmbed.from_torch(mesh_device, embeddings.proj, dtype=ttnn.bfloat16)
    out = to_host(tt_patch_embed(rows_to_device(x_in, mesh_device)), mesh_device, ref_out.shape)

    assert_close(
        f"patch_embed {grid_name(grid_thw)} L={x_in.shape[0]}", ref_out, out, 0.9999, MAX_REL_L2["patch_embed"]
    )
