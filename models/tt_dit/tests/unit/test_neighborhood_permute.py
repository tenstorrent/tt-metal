# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for Natural <-> Bricked token order.

The load-bearing test is :func:`test_bricked_order_matches_index_formula`. The kernel side
computes addresses from ``site_to_bricked_index`` in ``neighborhood_plan.cpp``; this module
realises the same ordering by moving axes. Nothing in the type system ties those two
together, so if they ever disagree the kernel reads the wrong keys and still produces
plausible-looking video. That test is the pin.
"""

import math

import pytest
import torch

import ttnn
from models.tt_dit.layers.neighborhood_permute import (
    _TO_BRICKED_ORDER,
    SITES_PER_BRICK,
    brick_count,
    brick_grid,
    padded_volume,
    to_bricked,
    to_bricked_grid,
    to_natural,
)

LTX_STAGE5_VOLUME = (25, 272, 480)
CUBIC_BRICK = (2, 4, 4)
FLAT_TIME_BRICK = (1, 4, 8)


def reference_bricked_index_table(volume, brick):
    """The permutation, transcribed from ``site_to_bricked_index`` in neighborhood_plan.cpp.

    Deliberately written as index arithmetic rather than as axis moves, so it cannot share a
    bug with the reshape/permute implementation it checks.
    """
    volume_time, volume_height, volume_width = volume
    brick_time, brick_height, brick_width = brick
    bricks_height = volume_height // brick_height
    bricks_width = volume_width // brick_width

    table = torch.empty(volume_time * volume_height * volume_width, dtype=torch.long)
    for site_time in range(volume_time):
        for site_height in range(volume_height):
            for site_width in range(volume_width):
                brick_index = (
                    (site_time // brick_time) * (bricks_height * bricks_width)
                    + (site_height // brick_height) * bricks_width
                    + (site_width // brick_width)
                )
                site_index_in_brick = (
                    (site_time % brick_time) * (brick_height * brick_width)
                    + (site_height % brick_height) * brick_width
                    + (site_width % brick_width)
                )
                natural_index = (site_time * volume_height + site_height) * volume_width + site_width
                table[brick_index * SITES_PER_BRICK + site_index_in_brick] = natural_index
    return table


# ---------------------------------------------------------------------------
# Host-only
# ---------------------------------------------------------------------------


def test_padded_volume_rounds_up_to_whole_bricks():
    assert padded_volume((8, 12, 12), CUBIC_BRICK) == (8, 12, 12)
    # LTX stage 5 has 25 frames: a cubic brick needs one ghost frame, a flat one needs none.
    assert padded_volume(LTX_STAGE5_VOLUME, CUBIC_BRICK) == (26, 272, 480)
    assert padded_volume(LTX_STAGE5_VOLUME, FLAT_TIME_BRICK) == (25, 272, 480)


def test_brick_count_matches_padded_volume():
    assert brick_count((8, 12, 12), CUBIC_BRICK) == (8 // 2) * (12 // 4) * (12 // 4)
    assert brick_count(LTX_STAGE5_VOLUME, CUBIC_BRICK) == 13 * 68 * 120
    assert brick_count(LTX_STAGE5_VOLUME, FLAT_TIME_BRICK) == 25 * 68 * 60
    assert brick_grid((8, 12, 12), CUBIC_BRICK) == (4, 3, 3)
    assert brick_grid(LTX_STAGE5_VOLUME, CUBIC_BRICK) == (13, 68, 120)


def torch_to_bricked_grid(natural: torch.Tensor, brick: tuple[int, int, int]) -> torch.Tensor:
    """Host twin of ``to_bricked_grid`` for volumes that already divide into whole bricks."""
    batch, time_extent, height_extent, width_extent, channels = natural.shape
    brick_time, brick_height, brick_width = brick
    bricks_t, bricks_h, bricks_w = brick_grid((time_extent, height_extent, width_extent), brick)
    split = natural.reshape(batch, bricks_t, brick_time, bricks_h, brick_height, bricks_w, brick_width, channels)
    return split.permute(*_TO_BRICKED_ORDER).reshape(batch, bricks_t, bricks_h, bricks_w, SITES_PER_BRICK * channels)


def test_grid_form_is_a_reshape_of_the_flat_form():
    """``(b, T_br, H_br, W_br, 32*C)`` is the same memory as ``(b, sites, C)``.

    That reshape is the whole enabler: T_br and W_br stay genuine axes, so a halo exchange
    and a T-band slice do not need another permute.
    """
    brick = (8, 2, 2)
    volume = (16, 8, 8)
    channels = 4
    natural = torch.arange(math.prod(volume) * channels, dtype=torch.float32).reshape(1, *volume, channels)
    grid = torch_to_bricked_grid(natural, brick)
    flat = grid.reshape(1, brick_count(volume, brick) * SITES_PER_BRICK, channels)
    expected = natural.reshape(1, math.prod(volume), channels)[:, reference_bricked_index_table(volume, brick)]
    assert torch.equal(flat, expected)


def test_pad_w_br_matches_pad_natural_then_brick():
    """Halo on W_br after bricking is the same sites as padding W then bricking.

    Two W-shards of 8 columns, brick width 2, halo of 2 sites = 1 brick. The left shard
    plus the right shard's first W-brick equals the widened-then-bricked left region.
    """
    brick = (8, 2, 2)
    time_extent, height_extent, width_local = 8, 8, 8
    channels = 2
    width_full = width_local * 2
    natural = torch.arange(time_extent * height_extent * width_full * channels, dtype=torch.float32).reshape(
        1, time_extent, height_extent, width_full, channels
    )
    halo_sites = 2
    halo_bricks = halo_sites // brick[2]

    left = natural[:, :, :, :width_local, :]
    widened = natural[:, :, :, : width_local + halo_sites, :]
    bricked_widened = torch_to_bricked_grid(widened, brick)

    left_grid = torch_to_bricked_grid(left, brick)
    right = natural[:, :, :, width_local:, :]
    right_grid = torch_to_bricked_grid(right, brick)
    bricked_then_padded = torch.cat([left_grid, right_grid[:, :, :, :halo_bricks, :]], dim=3)

    assert torch.equal(bricked_then_padded, bricked_widened)


def test_t_br_slice_matches_slice_then_brick():
    """A T-range is a contiguous T_br slice iff its bounds are multiples of brick-T."""
    brick = (8, 2, 2)
    volume = (24, 8, 8)
    channels = 2
    natural = torch.arange(math.prod(volume) * channels, dtype=torch.float32).reshape(1, *volume, channels)
    lo, hi = 8, 16  # one T-brick
    sliced_then_bricked = torch_to_bricked_grid(natural[:, lo:hi], brick)
    bricked = torch_to_bricked_grid(natural, brick)
    t_br_lo, t_br_hi = lo // brick[0], hi // brick[0]
    bricked_then_sliced = bricked[:, t_br_lo:t_br_hi]
    assert torch.equal(bricked_then_sliced, sliced_then_bricked)


def test_bands_align_to_brick_t():
    """Pad and layout bounds are multiples of brick-T; last interior may keep the true T."""
    from models.tt_dit.models.vae.diffvae_ltx_stage5 import _bands

    bands = _bands(78, frames=73, kernel=11, align=8)
    assert bands[0].lo % 8 == 0
    for band in bands:
        assert band.pad_lo % 8 == 0
        assert band.pad_hi % 8 == 0
        assert band.layout_hi % 8 == 0
    assert bands[-1].hi == 78
    assert bands[-1].layout_hi == 80
    # Radius 5 rounds up to 8, so the first band's pad is a whole extra brick, not 5 frames.
    assert bands[0].pad_lo == 0


# ---------------------------------------------------------------------------
# On device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "volume, brick",
    [
        ((4, 8, 8), CUBIC_BRICK),
        ((2, 8, 16), FLAT_TIME_BRICK),
    ],
    ids=["cubic_brick", "flat_time_brick"],
)
def test_bricked_order_matches_index_formula(mesh_device, volume, brick):
    """Every site must land where site_to_bricked_index says it does.

    Each site carries its own natural index as its value. The volumes here hold at most 256
    sites so every index is exact in bfloat16 and the comparison can be equality, not a
    tolerance.
    """
    channel_count = 32
    site_count = volume[0] * volume[1] * volume[2]
    assert site_count <= 256, "indices must be exactly representable in bfloat16"

    natural = (
        torch.arange(site_count, dtype=torch.float32)
        .reshape(1, *volume, 1)
        .expand(1, *volume, channel_count)
        .contiguous()
    )
    on_device = ttnn.from_torch(natural, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)

    actual = ttnn.to_torch(to_bricked(on_device, volume=volume, brick=brick)).float()[0, :, 0]
    expected = natural.reshape(1, site_count, channel_count)[0, reference_bricked_index_table(volume, brick), 0]

    assert torch.equal(actual, expected), "bricked order disagrees with site_to_bricked_index"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "volume, brick",
    [
        ((8, 12, 12), CUBIC_BRICK),
        ((25, 8, 12), CUBIC_BRICK),  # time not a brick multiple: exercises the ghost frame
        ((6, 8, 16), FLAT_TIME_BRICK),
    ],
    ids=["divides_evenly", "needs_ghost_frame", "flat_time_brick"],
)
def test_round_trip_is_identity(mesh_device, volume, brick):
    channel_count = 64
    natural = torch.randn(1, *volume, channel_count)
    on_device = ttnn.from_torch(natural, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)

    bricked = to_bricked(on_device, volume=volume, brick=brick)
    assert tuple(bricked.shape) == (1, brick_count(volume, brick) * SITES_PER_BRICK, channel_count)

    recovered = ttnn.to_torch(to_natural(bricked, volume=volume, brick=brick)).float()
    assert tuple(recovered.shape) == (1, *volume, channel_count)
    # Pure data movement: bfloat16 round-trips exactly, so this is equality not allclose.
    assert torch.equal(recovered, natural.bfloat16().float())


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
def test_bricked_grid_flattens_to_to_bricked(mesh_device):
    """The 5-D grid form is a view of the same sites ``to_bricked`` produces."""
    volume, brick = (8, 12, 12), CUBIC_BRICK
    channel_count = 32
    natural = torch.randn(1, *volume, channel_count)
    on_device = ttnn.from_torch(natural, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    grid = to_bricked_grid(on_device, volume=volume, brick=brick)
    bricks_t, bricks_h, bricks_w = brick_grid(volume, brick)
    assert tuple(grid.shape) == (1, bricks_t, bricks_h, bricks_w, SITES_PER_BRICK * channel_count)
    flat = ttnn.reshape(grid, (1, brick_count(volume, brick) * SITES_PER_BRICK, channel_count))
    expected = ttnn.to_torch(to_bricked(on_device, volume=volume, brick=brick)).float()
    assert torch.equal(ttnn.to_torch(flat).float(), expected)


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
def test_rejects_tile_layout(mesh_device, expect_error):
    """TILE layout pads each brick extent out to 32 and runs ~15x slower; refuse it loudly."""
    tensor = ttnn.from_torch(
        torch.randn(1, 4, 8, 8, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    with expect_error(ValueError, "ROW_MAJOR"):
        to_bricked(tensor, volume=(4, 8, 8), brick=CUBIC_BRICK)
