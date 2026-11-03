# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end accuracy test for M3VLVisionModel (tower, then projector) vs the Torch reference.

Two checks per grid, from the same pixel values:
  - forward_tower (patch embed, pre_layrnorm, NUM_LAYERS blocks) vs the reference tower output;
  - forward (tower and projector) vs the reference vision tokens.
A second test gives one model images of several grids with the same patch count, which it runs on the programs it
compiled for the first, each with the RoPE tables of its own grid. A third runs one tower with its activations in L1,
with and without the MLP's hidden activations in block shards, and in DRAM, and compares the outputs bit for bit.

Random weights and pixel values (tests/conftest.py), no checkpoint. The tests run NUM_LAYERS = 2 layers, and the L1
against DRAM test 3. The checkpoint has 32; that depth is for a real-weight test, since random weights say little
about error built up over 32 layers.

Run (the second line adds the other grids of VISION_GRIDS to CI_GRIDS, and of L1_VS_DRAM_ALL_GRIDS to
L1_VS_DRAM_CI_GRIDS, in vision_helpers.py):
    pytest models/demos/minimax_m3/tests/unit/test_vision_tower_vs_ref.py
    M3_VISION_ALL_GRIDS=1 pytest models/demos/minimax_m3/tests/unit/test_vision_tower_vs_ref.py
"""
from __future__ import annotations

import math

import torch
from loguru import logger

import ttnn
from models.demos.minimax_m3.tt.vision.block import M3VLBlock
from models.demos.minimax_m3.tt.vision.common import (
    ACTIVATIONS,
    DRAM_PLACEMENT,
    block_placement,
    l1_free_for_activations,
)
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import (
    L1_VS_DRAM_ALL_GRIDS,
    L1_VS_DRAM_CI_GRIDS,
    MAX_REL_L2,
    assert_close,
    grid_name,
    parametrize_vision_grids,
    to_host,
)

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

    model = M3VLVisionModel.from_torch(mesh_device, vision_reference(NUM_LAYERS), dtype=ttnn.bfloat16)
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
    image stages them again. Each forward copies its grid's tables to L1 for the blocks, which keep their activations
    in L1, as block_placement and the check of L1 before each forward confirm for every image; one more small L1
    tensor, allocated before each image after the first and kept, puts those copies and the activations at other
    addresses than the image before's."""
    monkeypatch.setattr("models.demos.minimax_m3.tt.vision.model.ROPE_CACHE_GRIDS", 2)
    model = M3VLVisionModel.from_torch(
        mesh_device, vision_reference(NUM_LAYERS), with_projector=False, dtype=ttnn.bfloat16
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
        name = f"tower {grid_name(grid_thw)} layers={NUM_LAYERS}, image {i + 1} of {len(grids)} on one model"
        placement = block_placement(model.blocks[0], math.prod(grid_thw), model.activations)
        assert placement is not DRAM_PLACEMENT, f"{name}: block_placement keeps the activations in DRAM"
        free, why_not = l1_free_for_activations(mesh_device)
        assert free, f"{name}: the blocks would keep their activations in DRAM: {why_not}"
        acts = vision_activations(grid_thw, NUM_LAYERS)
        tower_ref = acts["tower.out"]
        tower_tt = to_host(model.forward_tower(acts["pixel_values"], acts["grid_thw"]), mesh_device, tower_ref.shape)
        assert_close(name, tower_ref, tower_tt, 0.9998, MAX_REL_L2["tower"])
        if cache_entries is None:
            cache_entries = mesh_device.num_program_cache_entries()
            assert cache_entries > 0, "the program cache holds no programs after the first image"
        assert mesh_device.num_program_cache_entries() == cache_entries, f"{name} added a program to the cache"


def _l1_state(mesh_device):
    """The L1 bytes allocated per bank and the largest free block, as ttnn's memory view gives them."""
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    return view.total_bytes_allocated_per_bank, view.largest_contiguous_bytes_free_per_bank


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids(L1_VS_DRAM_CI_GRIDS, L1_VS_DRAM_ALL_GRIDS)
def test_vision_tower_l1_vs_dram(mesh_device, device_params, grid_thw, vision_reference):
    """A 3-block tower, run on one upload with each of ACTIVATIONS, gives the same output bit for bit with "l1" and
    "l1_interleaved" as with "dram": block 0 and both layouts the later blocks take in L1, as their input alternates
    between two addresses. Each forward takes block_placement's placement, which it logs, and not the DRAM fallback of
    the check of L1 before each forward, so a smaller L1 bank fails here rather than slowing the encoder. At the grids
    of L1_VS_DRAM_CI_GRIDS both L1 placements keep the activations in L1, and at h36w36 "l1" keeps the MLP's hidden
    activations in block shards. Of the other grids of L1_VS_DRAM_ALL_GRIDS, which M3_VISION_ALL_GRIDS=1 adds, h18w18
    keeps them in DRAM, and the rest put the MLP's shards in every layout the rule admits on 11x10, 12x10 and 13x10.
    Each forward leaves L1 as it found it, so the ops after the tower, the projector's, see the free L1 they saw
    before."""
    reference = vision_reference()
    model = M3VLVisionModel.from_torch(mesh_device, reference, with_projector=False, dtype=ttnn.bfloat16)
    # A third block, with layer 0's weights: the test compares the two placements, not against the reference.
    model.blocks.append(
        M3VLBlock.from_torch(mesh_device, reference.layers[0], model.padded_head_dim, dtype=ttnn.bfloat16)
    )
    rows, hidden = math.prod(grid_thw), reference.config.hidden_size
    generator = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(rows, reference.config.patch_dim, generator=generator)
    grid = torch.tensor([grid_thw])
    x_tt = model.push_pixel_values(pixel_values, grid)
    name = f"{grid_name(grid_thw)} layers={len(model.blocks)}"
    # "dram" first, the output the others must equal.
    modes = sorted(ACTIVATIONS, key=lambda activations: activations != "dram")
    placements = {
        activations: block_placement(model.blocks[0], x_tt.padded_shape[-2], activations) for activations in modes
    }
    logger.info(
        f"tower {name}: " + ", ".join(f"activations {a} take placement {p.name}" for a, p in placements.items())
    )
    if grid_name(grid_thw) in L1_VS_DRAM_CI_GRIDS:
        in_dram = [activations for activations in modes[1:] if placements[activations] is DRAM_PLACEMENT]
        assert not in_dram, f"{name}: block_placement keeps the activations in DRAM with {in_dram}"

    outputs = {}
    for activations in modes:
        model.activations = activations
        taken = model.placement(x_tt)
        assert taken is placements[activations], f"{name}: activations {activations} took placement {taken.name}"
        before = _l1_state(mesh_device)
        out = model.forward_device(x_tt, grid)
        after = _l1_state(mesh_device)
        assert after == before, f"{name}: activations {activations} left L1 at {after}, from {before}"
        outputs[activations] = to_host(out, mesh_device, (rows, hidden))
        ttnn.deallocate(out)
    for activations in modes[1:]:
        out = outputs[activations]
        num_bad = int((~torch.isfinite(out)).sum())
        assert not num_bad, f"{name}: {num_bad} of {out.numel()} output values are NaN or Inf with {activations}"
        num_diff = int((out.view(torch.int32) != outputs["dram"].view(torch.int32)).sum())
        assert not num_diff, f"{name}: {num_diff} of {out.numel()} output values differ between {activations} and dram"
