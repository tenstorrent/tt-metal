# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for M3VLProjector vs the Torch reference.

The projector takes the tower output (L, 1280) and makes the vision tokens (L / 4, 6144): a per-patch
MLP, then each 2x2 block of patches folded into one row and a second MLP. Random weights and pixel
values (conftest.py), no checkpoint.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_projector_vs_ref.py
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.projector import M3VLProjector

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import assert_pcc, grid_name, parametrize_vision_grids, to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_projector_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Projector on the reference tower output matches the reference vision tokens at PCC >= 0.99."""
    acts = vision_activations(grid_thw)
    x_in, ref_out = acts["tower.out"], acts["final"]
    reference = vision_reference()

    tt_projector = M3VLProjector.from_torch(
        mesh_device, reference.projector, spatial_merge_size=reference.config.spatial_merge_size, dtype=ttnn.bfloat16
    )
    out = to_host(tt_projector(to_device(x_in, mesh_device)), mesh_device, ref_out.shape)

    assert_pcc(f"projector {grid_name(grid_thw)} L={x_in.shape[0]}->{ref_out.shape[0]}", ref_out, out, 0.99)
