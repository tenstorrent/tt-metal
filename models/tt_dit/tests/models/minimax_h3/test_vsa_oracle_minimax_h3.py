# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Pin the VSA torch oracle to upstream FastVideo semantics (CPU-only).

``vsa_oracle`` computes fine attention by gathering each row's listed blocks;
upstream computes it as dense SDPA under an expanded boolean mask. The two are
independent computations of the same contract, so agreement here validates the
oracle that R3/R4/R6 device tests will trust.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ....models.transformers.minimax_h3.vsa_reference.loader import load_upstream
from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, build_vsa_geometry
from .vsa_oracle import (
    VSA_INDEX_SENTINEL,
    coarse_output,
    coarse_scores,
    fine_attention,
    select_index_rows,
    vsa_attention,
)

_TINY64 = ((70, 5, 130), (9, 10, 13))


@pytest.fixture(scope="module")
def upstream():
    base, h3 = load_upstream()
    # the vendored upstream test module carries the token-level mask oracle
    spec = importlib.util.spec_from_file_location(
        "vsa_reference_test_vsa_h3_metadata",
        Path(__file__).parents[3] / "models/transformers/minimax_h3/vsa_reference/test_vsa_h3_metadata.py",
    )
    upstream_test = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream_test
    spec.loader.exec_module(upstream_test)
    return base, h3, upstream_test


def _tiled_qkv(geometry, heads=2, dim=8, seed=0):
    torch.manual_seed(seed)
    q, k, v = (torch.randn(geometry.seq_len, heads, dim) for _ in range(3))
    # [S, H, D] -> tiled [1, H, S_pad, D], zeros at pad slots
    tile = lambda x: geometry.pack_rows(x).permute(1, 0, 2).unsqueeze(0)
    return (q, k, v), (tile(q), tile(k), tile(v))


@pytest.mark.parametrize("sparsity", [0.0, 0.75])
def test_index_sets_match_upstream_mask(upstream, sparsity):
    _, h3, _ = upstream
    prefix_segments, grid = _TINY64
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=1)
    _, (tq, tk, _) = _tiled_qkv(geometry)

    scores = coarse_scores(tq, tk, geometry)
    indices, sets = select_index_rows(scores, geometry, sparsity)
    mask = h3._build_block_mask(scores, geometry.n_prefix_tiles, geometry.n_video_tiles, sparsity, exempt=True)

    for h in range(scores.shape[1]):
        for row in range(geometry.n_tiles):
            expected = set(torch.nonzero(mask[0, h, row], as_tuple=False).reshape(-1).tolist())
            assert sets[h][row] == expected, (h, row)

    # sentinel tails are contiguous and the tensor is static-width uint32
    assert indices.dtype == torch.uint32
    flat = indices.to(torch.int64).reshape(-1, geometry.n_tiles)
    is_sentinel = flat == VSA_INDEX_SENTINEL
    first = is_sentinel.int().argmax(dim=-1)
    for row, start in enumerate(first.tolist()):
        if bool(is_sentinel[row].any()):
            assert bool(is_sentinel[row, start:].all()), row


@pytest.mark.parametrize("sparsity", [0.0, 0.75])
def test_fine_attention_matches_upstream_reference(upstream, sparsity):
    _, h3, upstream_test = upstream
    prefix_segments, grid = _TINY64
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=1)
    _, (tq, tk, tv) = _tiled_qkv(geometry, seed=1)

    scores = coarse_scores(tq, tk, geometry)
    indices, _ = select_index_rows(scores, geometry, sparsity)
    ours = fine_attention(tq, tk, tv, indices, geometry.valid_counts.to(torch.uint32))

    mask = h3._build_block_mask(scores, geometry.n_prefix_tiles, geometry.n_video_tiles, sparsity, exempt=True)
    meta = _upstream_meta(geometry)
    theirs_bshd = upstream_test.reference_sparse_attention(
        tq.permute(0, 2, 1, 3), tk.permute(0, 2, 1, 3), tv.permute(0, 2, 1, 3), mask, meta
    )
    theirs = theirs_bshd.permute(0, 2, 1, 3)

    valid = geometry.gather_index >= 0
    assert torch.allclose(ours[:, :, valid], theirs[:, :, valid], atol=1e-5), (
        (ours - theirs)[:, :, valid].abs().max()
    )


def _upstream_meta(geometry):
    """Duck-typed stand-in for upstream metadata: the reference reads two fields."""

    class _Meta:
        variable_block_sizes = geometry.valid_counts
        tile_elems = VSA_TILE_TOKENS

    return _Meta()


def test_sparsity_zero_unpacks_to_dense_sdpa():
    prefix_segments, grid = _TINY64
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=1)
    (q, k, v), (tq, tk, tv) = _tiled_qkv(geometry, seed=2)

    out_tiled = vsa_attention(tq, tk, tv, geometry, sparsity=0.0)
    out = geometry.unpack_rows(out_tiled.squeeze(0).permute(1, 0, 2))  # [S, H, D]

    dense = F.scaled_dot_product_attention(
        q.permute(1, 0, 2).unsqueeze(0), k.permute(1, 0, 2).unsqueeze(0), v.permute(1, 0, 2).unsqueeze(0)
    ).squeeze(0).permute(1, 0, 2)
    assert torch.allclose(out, dense, atol=1e-5), (out - dense).abs().max()


def test_gate_branch_matches_upstream(upstream):
    _, h3, _ = upstream
    prefix_segments, grid = _TINY64
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=1)
    _, (tq, tk, tv) = _tiled_qkv(geometry, seed=3)
    torch.manual_seed(4)
    gate = torch.randn_like(tq)

    scores = coarse_scores(tq, tk, geometry)
    ours = gate * coarse_output(scores, tv, geometry, tq.dtype)

    # upstream gate branch (video_sparse_attn_h3.forward L750-765) on BSHD
    v_pooled = h3._pool_tiles(tv.permute(0, 2, 1, 3), geometry.valid_counts, VSA_TILE_TOKENS)
    out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)  # [B, H, n_tiles, D]
    theirs = gate.permute(0, 2, 1, 3).reshape(
        1, geometry.n_tiles, VSA_TILE_TOKENS, *gate.shape[1:2], gate.shape[-1]
    ) * out_c.permute(0, 2, 1, 3).unsqueeze(2)
    theirs = theirs.reshape(1, geometry.padded_len, gate.shape[1], gate.shape[-1]).permute(0, 2, 1, 3)

    assert torch.allclose(ours, theirs, atol=1e-5), (ours - theirs).abs().max()


def test_striped_placement_same_output_after_unpack():
    """R6d at oracle level: striped and identity placements agree after unpacking."""
    prefix_segments, grid = _TINY64
    identity = build_vsa_geometry(prefix_segments, grid, sp_factor=8, placement="identity")
    striped = build_vsa_geometry(prefix_segments, grid, sp_factor=8, placement="striped")

    torch.manual_seed(5)
    q, k, v, g = (torch.randn(identity.seq_len, 2, 8) for _ in range(4))
    outs = {}
    for name, geometry in (("identity", identity), ("striped", striped)):
        tile = lambda x: geometry.pack_rows(x).permute(1, 0, 2).unsqueeze(0)
        out_tiled = vsa_attention(tile(q), tile(k), tile(v), geometry, sparsity=0.75, gate_tiled=tile(g))
        outs[name] = geometry.unpack_rows(out_tiled.squeeze(0).permute(1, 0, 2))
    assert torch.allclose(outs["identity"], outs["striped"], atol=1e-5), (
        (outs["identity"] - outs["striped"]).abs().max()
    )
