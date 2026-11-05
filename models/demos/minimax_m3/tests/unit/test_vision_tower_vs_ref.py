# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end accuracy test for M3VLVisionModel (tower, then projector) vs Hugging Face's MiniMaxM3VLVisionModel and
MiniMaxM3VLMultiModalProjector, the model built with from_state_dict from the reference's weights.

Two checks per grid, from the same pixel values:
  - forward_tower (patch embed, pre_layrnorm, NUM_LAYERS blocks) vs the reference tower output;
  - forward (tower and projector) vs the reference vision tokens.
A second test gives one model images of several grids with the same patch count, which it runs on the programs it
compiled for the first, each with the RoPE tables of its own grid.

Random weights and pixel values (tests/conftest.py), no checkpoint. The tests run NUM_LAYERS = 2 layers. The
checkpoint has 32; that depth is for a real-weight test, since random weights say little about error built up over 32
layers.

Run (the second line adds the other grids of VISION_GRIDS to CI_GRIDS, both in vision_helpers.py):
    pytest models/demos/minimax_m3/tests/unit/test_vision_tower_vs_ref.py
    M3_VISION_ALL_GRIDS=1 pytest models/demos/minimax_m3/tests/unit/test_vision_tower_vs_ref.py
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import MAX_REL_L2, assert_close, grid_name, parametrize_vision_grids, to_host

NUM_LAYERS = 2  # of the checkpoint's 32


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
def test_vision_tower_vs_ref(mesh_device, device_params, grid_thw, vision_reference, vision_activations):
    """Tower output and vision tokens match the reference at PCC >= 0.9998, within their relL2 bounds and with no NaN
    or Inf."""
    acts = vision_activations(grid_thw, NUM_LAYERS)
    pixel_values, grid = acts["pixel_values"], acts["grid_thw"]
    tower_ref, final_ref = acts["tower.out"], acts["final"]

    reference = vision_reference(NUM_LAYERS)
    model = M3VLVisionModel.from_state_dict(mesh_device, reference.weights, reference.config, dtype=ttnn.bfloat16)
    name = f"{grid_name(grid_thw)} layers={NUM_LAYERS}"

    tower_tt = to_host(model.forward_tower(pixel_values, grid), mesh_device, tower_ref.shape)
    assert_close(f"tower {name}", tower_ref, tower_tt, 0.9998, MAX_REL_L2["tower"])

    final_tt = to_host(model.forward(pixel_values, grid), mesh_device, final_ref.shape)
    assert_close(f"final {name}", final_ref, final_tt, 0.9998, MAX_REL_L2["final"])


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_vision_tower_new_grid_on_cached_programs(
    mesh_device, device_params, monkeypatch, vision_reference, vision_activations
):
    """One model takes images of h16w16, h8w32, h32w8 and h16w16 again, 256 patches each: each tower output matches
    the reference of its own grid, and no image after the first adds a program to the cache. The model keeps the RoPE
    tables of two grids here, so h8w32's tables are staged while h16w16's are still allocated, and a program that kept
    the first tables' addresses would rotate h8w32 by h16w16's positions; h32w8 frees h16w16's tables, and the last
    image stages them again. Each forward copies its grid's tables to L1 for the blocks; one more small L1 tensor,
    allocated before each image after the first and kept, puts those copies at other addresses than the image
    before's."""
    monkeypatch.setattr("models.demos.minimax_m3.tt.vision.model.ROPE_CACHE_GRIDS", 2)
    reference = vision_reference(NUM_LAYERS)
    model = M3VLVisionModel.from_state_dict(
        mesh_device, reference.weights, reference.config, with_projector=False, dtype=ttnn.bfloat16
    )
    grids = ((1, 16, 16), (1, 8, 32), (1, 32, 8), (1, 16, 16))
    cache_entries, held = None, []
    for i, grid_thw in enumerate(grids):
        if i:
            held.append(
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.L1_MEMORY_CONFIG
                )
            )
        acts = vision_activations(grid_thw, NUM_LAYERS)
        tower_ref = acts["tower.out"]
        tower_tt = to_host(model.forward_tower(acts["pixel_values"], acts["grid_thw"]), mesh_device, tower_ref.shape)
        name = f"tower {grid_name(grid_thw)} layers={NUM_LAYERS}, image {i + 1} of {len(grids)} on one model"
        assert_close(name, tower_ref, tower_tt, 0.9998, MAX_REL_L2["tower"])
        if cache_entries is None:
            cache_entries = mesh_device.num_program_cache_entries()
            assert cache_entries > 0, "the program cache holds no programs after the first image"
        assert mesh_device.num_program_cache_entries() == cache_entries, f"{name} added a program to the cache"
