# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Device test of M3VLProjector against Hugging Face's MiniMaxM3VLMultiModalProjector, on the reference tower's output.

The weights are seeded random values (tests/conftest.py), so no checkpoint is needed.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.projector import M3VLProjector

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import MAX_REL_L2, assert_close, grid_name, parametrize_vision_grids, rows_to_device, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_projector_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Projector on the reference tower output matches the reference vision tokens at PCC >= 0.9999, within its relL2
    bound and with no NaN or Inf."""
    x_in = vision_activations(grid_thw)["tower.out"].bfloat16().float()
    projector = vision_reference().projector
    ref_out = projector(x_in)

    tt_projector = M3VLProjector.from_torch(mesh_device, projector, dtype=ttnn.bfloat16)
    out = to_host(tt_projector(rows_to_device(x_in, mesh_device)), mesh_device, ref_out.shape)

    name = f"projector {grid_name(grid_thw)} L={x_in.shape[0]}->{ref_out.shape[0]}"
    assert_close(name, ref_out, out, 0.9999, MAX_REL_L2["projector"])
