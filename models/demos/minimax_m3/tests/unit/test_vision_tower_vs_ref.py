# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Device tests of M3VLVisionModel, built with from_state_dict, against Hugging Face's vision tower and projector.

test_vision_tower_vs_ref compares the tower output and the vision tokens on each grid. The second test gives one model
images of several grids with the same patch count, which must reuse the first image's programs. Both run 2 layers with
seeded random weights (tests/conftest.py); test_vision_accuracy.py runs all 32 with real weights.
test_vision_tower_l1_vs_dram runs a 3-layer tower with its activations in L1, with and without the MLP's hidden
activations in block shards, and in DRAM. The outputs must be equal bit for bit.
M3_VISION_ALL_GRIDS=1 adds the other grids of VISION_GRIDS, and of L1_VS_DRAM_ALL_GRIDS for the L1 test. It also adds
test_vision_tower_l1_clash_falls_back_to_dram, which forces a buffer clash in L1 on the chip.
"""
from __future__ import annotations

import math
import os

import torch
from loguru import logger

import ttnn
from models.demos.minimax_m3.tt.vision import common
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
    VISION_GRIDS,
    assert_close,
    grid_name,
    parametrize_vision_grids,
    to_host,
)

NUM_LAYERS = 2  # of the checkpoint's 32
# Bytes per L1 bank that the clash test puts at the top of L1. With them, an op of the L1 placement at 2304 patches
# lacks over 100 KB for its buffers on every grid of L1_GRIDS. The DRAM placement need not fit beside all of them (on
# 12x10 the QKV matmul's buffers reach past them), so the test frees all but PROOF_L1_FOREIGN_BYTES, the most that the
# L1 check allows in use, when the L1 forward fails.
CLASH_FOREIGN_BYTES = 384 * 1024


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
    """One model runs h16w16, h8w32, h32w8 and h16w16 again, all of 256 patches, on the first image's programs.

    Each tower output matches its own grid's reference. The RoPE cache holds two grids here, so tables are freed and
    staged again. The blocks keep their activations in L1, which the test checks before each image. A small L1 tensor
    kept before each later image moves the tables' L1 copies and the activations, so a program that kept old addresses
    would fail.
    """
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
    """Return the L1 bytes allocated per bank and the largest free block, from ttnn's memory view."""
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    return view.total_bytes_allocated_per_bank, view.largest_contiguous_bytes_free_per_bank


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids(L1_VS_DRAM_CI_GRIDS, L1_VS_DRAM_ALL_GRIDS)
def test_vision_tower_l1_vs_dram(mesh_device, device_params, grid_thw, vision_reference):
    """A 3-block tower gives the same output bit for bit with "l1" and "l1_interleaved" as with "dram".

    Three blocks cover both L1 addresses that the later blocks' inputs alternate between. Each forward must take
    block_placement's placement, and at the CI grids that placement must use L1. So a smaller L1 bank fails here instead
    of slowing the encoder. Each forward must also leave L1 as it found it, and must not fall back to DRAM: a fallback
    means an op's buffers no longer fit beside the activations, for example after a tt-metal kernel change.
    """
    reference = vision_reference()
    model = M3VLVisionModel.from_state_dict(
        mesh_device, reference.weights, reference.config, with_projector=False, dtype=ttnn.bfloat16
    )
    # A third block, with layer 0's weights: the test compares the two placements, not against the reference.
    model.blocks.append(
        M3VLBlock.from_torch(mesh_device, reference.tower.layers[0], model.padded_head_dim, dtype=ttnn.bfloat16)
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
        before, fallbacks = _l1_state(mesh_device), model._dram_fallbacks
        out = model.forward_device(x_tt, grid)
        after = _l1_state(mesh_device)
        assert after == before, f"{name}: activations {activations} left L1 at {after}, from {before}"
        fell_back = model._dram_fallbacks != fallbacks
        assert not fell_back, f"{name}: activations {activations} fell back to DRAM: {model._warned_fallback}"
        outputs[activations] = to_host(out, mesh_device, (rows, hidden))
        ttnn.deallocate(out)
    for activations in modes[1:]:
        out = outputs[activations]
        num_bad = int((~torch.isfinite(out)).sum())
        assert not num_bad, f"{name}: {num_bad} of {out.numel()} output values are NaN or Inf with {activations}"
        num_diff = int((out.view(torch.int32) != outputs["dram"].view(torch.int32)).sum())
        assert not num_diff, f"{name}: {num_diff} of {out.numel()} output values differ between {activations} and dram"


# Only in the full-grid run (M3_VISION_ALL_GRIDS=1): alone, its kernel builds take longer than the CI entry can spare.
# Otherwise it is left out rather than skipped, as the extra grids are.
if os.getenv("M3_VISION_ALL_GRIDS") == "1":

    @torch.no_grad()
    @parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
    def test_vision_tower_l1_clash_falls_back_to_dram(mesh_device, device_params, monkeypatch, vision_reference):
        """When an L1 forward hits tt-metal's buffer clash on the chip, the image runs again in DRAM.

        After a "dram" forward, tensors at the top of L1, which the L1 check is told to allow, force the clash at 2304
        patches. When the L1 attempt fails, the test frees all of them but PROOF_L1_FOREIGN_BYTES, the most that the L1
        check allows in use. In use the rerun gets its room when the failed run's own L1 tensors are freed, so the
        rerun must start with only the kept tensor in L1. The "l1" forward must log the clash, leave L1 as it found it
        and match the "dram" forward bit for bit.
        """
        grid = mesh_device.compute_with_storage_grid_size()
        reference = vision_reference()
        model = M3VLVisionModel.from_state_dict(
            mesh_device, reference.weights, reference.config, with_projector=False, dtype=ttnn.bfloat16
        )
        grid_thw = VISION_GRIDS["h48w48"]
        rows, hidden = math.prod(grid_thw), reference.config.hidden_size
        generator = torch.Generator().manual_seed(0)
        pixel_values = torch.randn(rows, reference.config.patch_dim, generator=generator)
        image_grid = torch.tensor([grid_thw])
        x_tt = model.push_pixel_values(pixel_values, image_grid)
        name = f"h48w48 layers={len(model.blocks)} on {grid.x}x{grid.y}"
        empty = _l1_state(mesh_device)
        model.activations = "dram"
        assert model.placement(x_tt) is DRAM_PLACEMENT, name
        out = model.forward_device(x_tt, image_grid)
        assert _l1_state(mesh_device) == empty, f"{name}: the dram forward left L1 at {_l1_state(mesh_device)}"
        want = to_host(out, mesh_device, (rows, hidden))
        ttnn.deallocate(out)

        def l1_tensor(bytes_per_bank):
            # Banks x tiles tiles of 2 KB, dealt round-robin over the banks: bytes_per_bank in each.
            tiles = bytes_per_bank // (32 * 32 * 2)
            shape = ttnn.Shape([1, 1, 32 * grid.x * grid.y, 32 * tiles])
            return ttnn.allocate_tensor_on_device(
                shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.L1_MEMORY_CONFIG
            )

        allowed = common.PROOF_L1_FOREIGN_BYTES
        kept = l1_tensor(allowed)  # allocated first, so at the very top of L1
        try:
            before = _l1_state(mesh_device)
            foreign = [l1_tensor(CLASH_FOREIGN_BYTES - allowed)]
            run_tower, entries = M3VLVisionModel._run_tower, []

            def run_tower_freeing_foreign(self, x, grid_thw, placement):
                entries.append(_l1_state(mesh_device))
                try:
                    return run_tower(self, x, grid_thw, placement)
                except RuntimeError:
                    if foreign:
                        ttnn.deallocate(foreign.pop())
                    raise

            monkeypatch.setattr(M3VLVisionModel, "_run_tower", run_tower_freeing_foreign)
            warnings = []
            sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
            try:
                monkeypatch.setattr(common, "PROOF_L1_FOREIGN_BYTES", 2 * CLASH_FOREIGN_BYTES)
                assert l1_free_for_activations(mesh_device) == (True, ""), name
                l1_placement = block_placement(model.blocks[0], x_tt.padded_shape[-2], "l1")
                assert l1_placement is not DRAM_PLACEMENT, name
                model.activations = "l1"
                assert model.placement(x_tt) is l1_placement, name
                out = model.forward_device(x_tt, image_grid)
                assert not foreign, f"{name}: the l1 forward did not clash, warnings {warnings}"
            finally:
                logger.remove(sink)
                if foreign:
                    ttnn.deallocate(foreign.pop())
            assert len(entries) == 2, f"{name}: {len(entries)} tower runs"
            assert entries[1] == before, f"{name}: the rerun started with L1 at {entries[1]}, not {before}"
            after = _l1_state(mesh_device)
            assert after == before, f"{name}: the l1 forward left L1 at {after}, from {before}"
            got = to_host(out, mesh_device, (rows, hidden))
            ttnn.deallocate(out)
        finally:
            ttnn.deallocate(kept)
        clashes = [line for line in warnings if "clash with L1 buffers" in line]
        assert len(clashes) == 1, f"{name}: warnings {warnings}"
        assert f"placement {l1_placement.name} at {rows} rows failed" in clashes[0], clashes[0]
        logger.info(f"tower {name}: {clashes[0]}")
        num_diff = int((got.view(torch.int32) != want.view(torch.int32)).sum())
        assert not num_diff, f"{name}: {num_diff} of {got.numel()} output values differ from dram"
