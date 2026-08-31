# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""R1 checks: the VSA tile geometry against genuine upstream FastVideo code.

CPU-only. The oracle is the vendored, untouched upstream
``_h3_tile_geometry`` (see ``vsa_reference/loader.py``); the hand-computed
fixtures mirror upstream's own ``test_vsa_h3_metadata.py`` so both suites pin
the same numbers.
"""

import math

import pytest
import torch

from ....models.transformers.minimax_h3.vsa_reference.loader import load_upstream
from ....pipelines.minimax_h3.vsa_geometry import (
    VSA_TILE_TOKENS,
    build_vsa_geometry,
    chop_prefix_segments,
    video_tile_partition_indices,
    video_tile_valid_counts,
)

_CPU = torch.device("cpu")

# (prefix_segments, video token grid). _TINY64 and _PROD mirror the upstream
# fixtures (upstream states them as raw latents over patch (1,2,2); these are
# the resulting token grids). _RAGGED9 covers ragged tails in all three dims;
# _15S768P is our production 15 s / 768p target: canvas 768x1344, 362 frames
# -> 107 latent frames, token grid (107, 24, 42), audio 600 latents x 2 ch.
_TINY64 = ((70, 5, 130), (9, 10, 13))
_PROD = ((300, 0, 414), (37, 24, 42))
_RAGGED9 = ((3, 64, 129), (5, 6, 7))
_15S768P = ((512, 1008, 1200), (107, 24, 42))

_FIXTURES = {
    "tiny64": _TINY64,
    "prod": _PROD,
    "ragged9": _RAGGED9,
    "15s768p": _15S768P,
}


@pytest.fixture(scope="module")
def upstream():
    return load_upstream()


@pytest.mark.parametrize("fixture", list(_FIXTURES), ids=list(_FIXTURES))
def test_matches_upstream_geometry(upstream, fixture):
    """Tile order, valid counts, and the packed-row map equal upstream's, bit for bit."""
    _, h3 = upstream
    prefix_segments, grid = _FIXTURES[fixture]
    upstream_prefix = tuple(s for s in prefix_segments if s > 0)

    (tile_partition_indices, variable_block_sizes, untile_combined_index, num_prefix_tiles, num_video_tiles) = (
        h3._h3_tile_geometry(upstream_prefix, grid, _CPU, (4, 4, 4))
    )

    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=1)
    assert geometry.n_prefix_tiles == num_prefix_tiles
    assert geometry.n_video_tiles == num_video_tiles
    assert geometry.n_pad_tiles == 0
    assert torch.equal(geometry.valid_counts, variable_block_sizes)
    assert torch.equal(geometry.untile_index, untile_combined_index)
    # gather_index is the inverse map: valid slots pull exactly the rows
    # upstream's tile partition lists, in the same order.
    valid = geometry.gather_index >= 0
    assert torch.equal(geometry.gather_index[valid], tile_partition_indices)


def test_tiny64_hand_computed_counts():
    """The hand-computed numbers from upstream's test_geometry_tile64_ragged_tails."""
    prefix_segments, grid = _TINY64
    assert chop_prefix_segments(prefix_segments) == [64, 6, 5, 64, 64, 2]
    t, h, w = grid
    counts = video_tile_valid_counts(grid)
    expected = torch.tensor(
        [
            min(4, t - 4 * tt) * min(4, h - 4 * hh) * min(4, w - 4 * ww)
            for tt in range(3)
            for hh in range(3)
            for ww in range(4)
        ],
        dtype=torch.long,
    )
    assert torch.equal(counts, expected)
    assert int(counts.min()) == 1 * 2 * 1  # the (t, h, w) ragged corner

    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=1)
    assert geometry.n_prefix_tiles == 6
    assert geometry.n_video_tiles == 3 * 3 * 4
    assert int(geometry.valid_counts.sum()) == geometry.seq_len

    # every packed video row lands in the 3D tile its (t, h, w) coordinate says
    prefix_len = sum(prefix_segments)
    idx = geometry.untile_index
    row = torch.arange(t * h * w)
    row_t, row_h, row_w = row // (h * w), (row // w) % h, row % w
    expected_tile = geometry.n_prefix_tiles + ((row_t // 4) * 3 + row_h // 4) * 4 + row_w // 4
    assert torch.equal(idx[prefix_len:] // VSA_TILE_TOKENS, expected_tile)
    assert bool((idx % VSA_TILE_TOKENS < geometry.valid_counts[idx // VSA_TILE_TOKENS]).all())


def test_segment_purity():
    """No prefix tile straddles a segment boundary (upstream test_geometry_720p's check)."""
    for prefix_segments, grid in _FIXTURES.values():
        geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=1)
        boundaries = torch.tensor(prefix_segments).cumsum(0).tolist()
        start = 0
        for size in geometry.valid_counts[: geometry.n_prefix_tiles].tolist():
            end = start + size
            assert all(not (start < b < end) for b in boundaries), (start, end)
            start = end


@pytest.mark.parametrize("placement", ["identity", "striped"])
@pytest.mark.parametrize("sp_factor", [1, 8])
def test_roundtrip_identity(placement, sp_factor):
    """unpack(pack(x)) == x for every fixture, placement, and SP factor."""
    for prefix_segments, grid in _FIXTURES.values():
        geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=sp_factor, placement=placement)
        assert geometry.n_tiles % sp_factor == 0
        x = torch.randn(geometry.seq_len, 3)
        packed = geometry.pack_rows(x)
        assert packed.shape[0] == geometry.padded_len
        assert torch.equal(geometry.unpack_rows(packed), x)
        # pad slots stay zero
        assert torch.equal(packed[geometry.gather_index < 0], torch.zeros_like(packed[geometry.gather_index < 0]))


def test_pad_tiles_and_shard_alignment():
    """Pad tiles top the count up to the SP factor and carry inert metadata."""
    prefix_segments, grid = _15S768P
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=8)
    n_real = geometry.n_prefix_tiles + geometry.n_video_tiles
    assert geometry.n_pad_tiles == math.ceil(n_real / 8) * 8 - n_real
    assert geometry.n_tiles % 8 == 0
    assert geometry.padded_len % (8 * VSA_TILE_TOKENS) == 0
    pad = geometry.tile_ids < 0
    assert int(pad.sum()) == geometry.n_pad_tiles
    assert not geometry.is_exempt[pad].any()
    assert not geometry.is_candidate[pad].any()
    assert (geometry.valid_counts[pad] == 0).all()
    # production scale sanity: ~1802 real tiles, k = ceil(0.1 * candidates) at 0.9 sparsity
    assert n_real == geometry.n_prefix_tiles + 27 * 6 * 11
    assert int(geometry.is_candidate.sum()) == 27 * 6 * 11


def test_striped_placement():
    """Striping permutes whole tiles, spreads exempt tiles, and unpacks identically."""
    for prefix_segments, grid in _FIXTURES.values():
        identity = build_vsa_geometry(prefix_segments, grid, sp_factor=8, placement="identity")
        striped = build_vsa_geometry(prefix_segments, grid, sp_factor=8, placement="striped")

        # same multiset of tiles: match up by canonical id
        order = torch.argsort(striped.tile_ids, stable=True)[striped.n_pad_tiles :]
        ident_order = torch.argsort(identity.tile_ids, stable=True)[identity.n_pad_tiles :]
        assert torch.equal(striped.tile_ids[order], identity.tile_ids[ident_order])
        assert torch.equal(striped.valid_counts[order], identity.valid_counts[ident_order])
        assert torch.equal(striped.is_exempt[order], identity.is_exempt[ident_order])

        # exempt tiles spread across shards: max/min per-shard counts differ by <= 1
        per_shard = striped.is_exempt.reshape(8, striped.tiles_per_shard).sum(dim=1)
        assert int(per_shard.max() - per_shard.min()) <= 1

        # both placements recover the same rows
        x = torch.randn(identity.seq_len, 2)
        assert torch.equal(striped.unpack_rows(striped.pack_rows(x)), x)
        assert torch.equal(identity.unpack_rows(identity.pack_rows(x)), x)


def test_averaging_matrix_matches_upstream_pooling(upstream):
    """A @ X equals upstream _pool_tiles (masked mean over zero-padded tiles)."""
    base, h3 = upstream
    torch.manual_seed(0)
    prefix_segments, grid = _TINY64
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=1)

    heads, dim = 2, 8
    x = torch.randn(geometry.seq_len, heads, dim)
    packed = geometry.pack_rows(x).unsqueeze(0)  # [1, S_pad, H, D], zeros at pads

    pooled_upstream = h3._pool_tiles(packed, geometry.valid_counts, VSA_TILE_TOKENS)  # [1, H, n_tiles, D]
    matrix = geometry.averaging_matrix()
    pooled_ours = torch.einsum("ts,bshd->bhtd", matrix, packed.to(torch.float32))
    assert torch.allclose(pooled_ours, pooled_upstream, atol=1e-5)


def test_row_source_replicates_tile_metadata():
    """permute_metadata keeps pad slots on a valid row of their own tile."""
    prefix_segments, grid = _TINY64
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=8)
    tags = torch.arange(geometry.seq_len)
    permuted = geometry.permute_metadata(tags)
    valid = geometry.gather_index >= 0
    assert torch.equal(permuted[valid], geometry.gather_index[valid])
    # pad slots inside a ragged tile replicate that tile's first row
    slot_tile = torch.arange(geometry.padded_len) // VSA_TILE_TOKENS
    for tile in range(geometry.n_tiles):
        count = int(geometry.valid_counts[tile])
        rows = permuted[slot_tile == tile]
        if count:
            assert torch.equal(rows[count:], rows[0].expand_as(rows[count:]))
