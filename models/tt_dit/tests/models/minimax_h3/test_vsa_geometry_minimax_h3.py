# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The VSA tile geometry against genuine upstream FastVideo code.

CPU-only. The oracle is the vendored, untouched upstream
``_h3_tile_geometry`` (see ``vsa_reference/loader.py``); the hand-computed
fixtures mirror upstream's own ``test_vsa_h3_metadata.py`` so both suites pin
the same numbers.
"""

import pytest
import torch

from ....pipelines.minimax_h3.vsa_geometry import (
    VSA_TILE_TOKENS,
    build_vsa_geometry,
    chop_prefix_segments,
    video_tile_valid_counts,
    vsa_min_padded_len,
)
from .vsa_reference.loader import load_upstream

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


def _geometry(prefix_segments, grid, sp_factor=1, extra_rows_per_shard=0):
    padded_len = vsa_min_padded_len(prefix_segments, grid, sp_factor) + sp_factor * extra_rows_per_shard
    return build_vsa_geometry(prefix_segments, grid, sp_factor=sp_factor, padded_len=padded_len)


@pytest.fixture(scope="module")
def upstream():
    return load_upstream()


@pytest.mark.parametrize("fixture", list(_FIXTURES), ids=list(_FIXTURES))
def test_matches_upstream_geometry(upstream, fixture):
    """Tile order, valid counts, and the packed-row map equal upstream's, bit for bit."""
    _, h3 = upstream
    prefix_segments, grid = _FIXTURES[fixture]
    upstream_prefix = tuple(s for s in prefix_segments if s > 0)

    (
        tile_partition_indices,
        variable_block_sizes,
        untile_combined_index,
        num_prefix_tiles,
        num_video_tiles,
    ) = h3._h3_tile_geometry(upstream_prefix, grid, _CPU, (4, 4, 4))

    geometry = _geometry(prefix_segments, grid)
    assert geometry.n_prefix_tiles == num_prefix_tiles
    assert geometry.n_video_tiles == num_video_tiles
    assert geometry.n_blocks == geometry.n_real_tiles
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
    # The (t, h, w) ragged corner.
    assert int(counts.min()) == 1 * 2 * 1

    geometry = _geometry(prefix_segments, grid)
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
        geometry = _geometry(prefix_segments, grid)
        boundaries = torch.tensor(prefix_segments).cumsum(0).tolist()
        start = 0
        for size in geometry.valid_counts[: geometry.n_prefix_tiles].tolist():
            end = start + size
            assert all(not (start < b < end) for b in boundaries), (start, end)
            start = end


@pytest.mark.parametrize("sp_factor", [1, 8])
@pytest.mark.parametrize("extra_rows_per_shard", [0, 32, 160])
def test_roundtrip_identity(sp_factor, extra_rows_per_shard):
    """unpack(pack(x)) == x for every fixture, SP factor and rung padding, stragglers included."""
    for prefix_segments, grid in _FIXTURES.values():
        geometry = _geometry(prefix_segments, grid, sp_factor, extra_rows_per_shard)
        x = torch.randn(geometry.seq_len, 3)
        packed = geometry.pack_rows(x)
        assert packed.shape[0] == geometry.padded_len
        assert torch.equal(geometry.unpack_rows(packed), x)
        pad_rows = geometry.gather_index < 0
        assert torch.equal(packed[pad_rows], torch.zeros_like(packed[pad_rows]))


def test_pad_blocks_in_the_rung(expect_error):
    """15 s / 768p at SP 8 with a straggler per shard: tiles fill whole blocks in order, the rest is inert."""
    prefix_segments, grid = _15S768P
    sp_factor = 8
    min_len = vsa_min_padded_len(prefix_segments, grid, sp_factor)
    with expect_error(ValueError, "fewer whole blocks"):
        build_vsa_geometry(prefix_segments, grid, sp_factor=sp_factor, padded_len=min_len - sp_factor * 32)
    # 15200 rows per shard: 237 whole blocks and a 32-row straggler.
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=sp_factor, padded_len=121600)
    assert geometry.rows_per_shard % VSA_TILE_TOKENS == 32

    real = geometry.valid_counts > 0
    assert int(real.sum()) == geometry.n_real_tiles
    pad = ~real
    assert not geometry.is_exempt[pad].any()
    assert not geometry.is_candidate[pad].any()
    straggler = torch.arange(geometry.n_blocks) % geometry.blocks_per_shard == geometry.blocks_per_shard - 1
    assert bool(pad[straggler].all())

    # Real tiles keep FastVideo order: the blocks holding them, in global order, carry the canonical counts.
    canonical = torch.cat([torch.tensor(chop_prefix_segments(prefix_segments)), video_tile_valid_counts(grid)])
    assert torch.equal(geometry.valid_counts[real], canonical)
    last_real = int(torch.nonzero(real).max())
    assert geometry.logical_n == int(geometry.block_starts[last_real]) + VSA_TILE_TOKENS
    assert int(real[last_real + 1 :].sum()) == 0

    # Production scale sanity: k = ceil(0.1 * candidates) at 0.9 sparsity.
    assert geometry.n_real_tiles == geometry.n_prefix_tiles + 27 * 6 * 11
    assert int(geometry.is_candidate.sum()) == 27 * 6 * 11


def test_averaging_matrix_matches_upstream_pooling(upstream):
    """A @ X equals upstream _pool_tiles (masked mean over zero-padded tiles)."""
    base, h3 = upstream
    torch.manual_seed(0)
    prefix_segments, grid = _TINY64
    geometry = _geometry(prefix_segments, grid)

    heads, dim = 2, 8
    x = torch.randn(geometry.seq_len, heads, dim)
    packed_bshd = geometry.pack_rows(x).unsqueeze(0)

    pooled_upstream = h3._pool_tiles(packed_bshd, geometry.valid_counts, VSA_TILE_TOKENS)
    matrix = geometry.averaging_matrix()
    pooled_ours = torch.einsum("ts,bshd->bhtd", matrix, packed_bshd.to(torch.float32))
    assert torch.allclose(pooled_ours, pooled_upstream, atol=1e-5)


def test_row_source_replicates_tile_metadata():
    """permute_metadata keeps pad rows on a valid row of their own tile, stragglers included."""
    prefix_segments, grid = _TINY64
    geometry = _geometry(prefix_segments, grid, sp_factor=8, extra_rows_per_shard=96)
    tags = torch.arange(geometry.seq_len)
    permuted = geometry.permute_metadata(tags)
    valid = geometry.gather_index >= 0
    assert torch.equal(permuted[valid], geometry.gather_index[valid])
    # Pad rows inside a ragged tile replicate that tile's first row.
    for block in range(geometry.n_blocks):
        count = int(geometry.valid_counts[block])
        rows = permuted[geometry.block_of_row == block]
        if count:
            assert torch.equal(rows[count:], rows[0].expand_as(rows[count:]))
