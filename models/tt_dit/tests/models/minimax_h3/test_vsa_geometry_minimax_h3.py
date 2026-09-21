# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the VSA tile geometry (no device)."""

from __future__ import annotations

import pytest
import torch

from models.tt_dit.pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, build_vsa_geometry

_GEOMETRY_FIELDS = (
    "valid_counts",
    "tile_ids",
    "is_3d",
    "is_exempt",
    "is_candidate",
    "gather_index",
    "untile_index",
    "row_source",
)


def _assert_same_tensors(a, b):
    for name in _GEOMETRY_FIELDS:
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_exact_fit_matches_uncapped():
    """Capacity params that leave no empty slot reproduce the uncapped geometry bit-for-bit."""
    prefix = (128,)
    grid = (4, 8, 4)
    uncapped = build_vsa_geometry(prefix, grid, sp_factor=2)
    exempt = int(uncapped.is_exempt.sum())
    capped = build_vsa_geometry(prefix, grid, sp_factor=2, exempt_tiles=exempt, total_tiles=uncapped.n_tiles)
    if uncapped.n_pad_tiles == 0:  # only a padless rung is bit-identical, k included
        assert capped.candidate_capacity == uncapped.candidate_capacity
    _assert_same_tensors(capped, uncapped)


def test_capacity_structure_is_request_independent():
    """Two different requests on one rung share every trace-baked structural quantity."""
    kw = dict(sp_factor=2, exempt_tiles=4, total_tiles=8, placement="interleaved")
    a = build_vsa_geometry((64,), (4, 4, 4), **kw)
    b = build_vsa_geometry((200,), (8, 4, 4), **kw)
    assert a.n_tiles == b.n_tiles == 8
    assert a.candidate_capacity == b.candidate_capacity == 4
    assert a.n_prefix_tiles == b.n_prefix_tiles == 4
    assert torch.equal(a.is_exempt, b.is_exempt)
    assert a.padded_len == b.padded_len == 8 * VSA_TILE_TOKENS


def test_capacity_masks_empty_candidate_slots():
    """Unfilled candidate slots are zero-valid and excluded from top-k candidacy."""
    g = build_vsa_geometry((64,), (4, 4, 4), sp_factor=2, exempt_tiles=4, total_tiles=8)
    assert int(g.is_candidate.sum()) == 1  # one real video tile
    assert g.candidate_capacity == 4  # but k ranges over all four candidate slots
    assert int((g.valid_counts == 0).sum()) == g.n_pad_tiles


def test_capacity_admission_raises(expect_error):
    with expect_error(ValueError, "exempt tiles"):
        build_vsa_geometry((64 * 5,), (4, 4, 4), sp_factor=2, exempt_tiles=2, total_tiles=8)
    with expect_error(ValueError, "candidate tiles"):
        build_vsa_geometry((64,), (16, 8, 8), sp_factor=2, exempt_tiles=4, total_tiles=8)
    with expect_error(ValueError, "multiple of sp_factor"):
        build_vsa_geometry((64,), (4, 4, 4), sp_factor=2, exempt_tiles=3, total_tiles=7)
    with expect_error(ValueError, "capacity pair"):
        build_vsa_geometry((64,), (4, 4, 4), sp_factor=2, exempt_tiles=4)


@pytest.mark.parametrize("placement", ["identity", "striped", "interleaved"])
def test_pack_unpack_round_trip(placement):
    prefix = (96, 64)
    grid = (4, 8, 4)
    g = build_vsa_geometry(prefix, grid, sp_factor=2, exempt_tiles=6, total_tiles=16, placement=placement)
    x = torch.randn(g.seq_len, 5)
    packed = g.pack_rows(x)
    assert packed.shape[0] == g.padded_len
    assert torch.equal(g.unpack_rows(packed), x)
