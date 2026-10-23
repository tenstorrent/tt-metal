# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end PCC test for M3VLVisionModel (tower, then projector) vs the Torch reference.

Two checks per grid, from the same pixel values:
  - forward_tower (patch embed, pre_layrnorm, num_layers blocks) vs the reference tower output;
  - forward (tower and projector) vs the reference vision tokens.

Random weights and pixel values (conftest.py), no checkpoint. CI runs 2 layers. The checkpoint has 32;
that depth is for a real-weight test, since random weights say little about error built up over 32 layers.

Run (the second line adds the 448x448, 672x672 and 640x480 grids to the CI grid, 224x224):
    pytest models/demos/minimax_m3/tests/unit/test_vision_tower_vs_ref.py
    M3_VISION_ALL_GRIDS=1 pytest models/demos/minimax_m3/tests/unit/test_vision_tower_vs_ref.py
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel
from models.demos.minimax_m3.tt.vision.model_config import MiniMaxM3VLModelArgs

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import assert_pcc, grid_name, parametrize_vision_grids, to_host


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("num_layers", [2], ids=lambda n: f"layers{n}")
@parametrize_vision_grids()
def test_vision_tower_vs_ref(mesh_device, device_params, grid_thw, num_layers, vision_reference, vision_activations):
    """Tower output and vision tokens match the reference at PCC >= 0.99."""
    acts = vision_activations(grid_thw, num_layers)
    pixel_values, grid = acts["pixel_values"], acts["grid_thw"]
    tower_ref, final_ref = acts["tower.out"], acts["final"]

    model_args = MiniMaxM3VLModelArgs(mesh_device=mesh_device, dtype=ttnn.bfloat16, num_hidden_layers=num_layers)
    model = M3VLVisionModel.from_torch(
        mesh_device, vision_reference(num_layers), model_args, with_projector=True, dtype=ttnn.bfloat16
    )
    name = f"{grid_name(grid_thw)} layers={num_layers}"

    tower_tt = to_host(model.forward_tower(pixel_values, grid), mesh_device, tower_ref.shape)
    assert_pcc(f"tower {name}", tower_ref, tower_tt, 0.99)

    final_tt = to_host(model.forward(pixel_values, grid), mesh_device, final_ref.shape)
    assert_pcc(f"final {name}", final_ref, final_tt, 0.99)
