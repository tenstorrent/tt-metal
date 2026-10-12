# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""SLA selection mode: build_sla_geometry and its block selection against LightX2V's get_block_map.

CPU-only. The oracle is a torch port of LightX2V ``lightx2v/common/ops/attn/utils/sla_util.py``
``get_block_map`` (smooth-K, per-block mean pooling with the tail block averaged over its valid rows,
unscaled scores, floor top-k); our side emulates MiniMaxH3VSACoarseStage's host math (averaging-matrix
pooling, scaled scores, -inf candidate mask over pad tiles, top-k of compute_topk_sla).
"""

import math

import pytest
import torch

from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import compute_topk_sla
from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, build_sla_geometry, sla_query_averaging_matrix

# (seq_len, video token grid): a small ragged case and 768p 10 s fl2va (text 512 + cond video 2 x 1008
# + audio 400 + 31 x 24 x 42 video rows; the grid only labels the geometry)
_CASES = {"tiny": (1000, (4, 6, 7)), "768p10s": (512 + 2016 + 400 + 31 * 24 * 42, (31, 24, 42))}


def _mean_pool(x: torch.Tensor, blk: int) -> torch.Tensor:
    """LightX2V compress_kernel: mean over each block's valid rows."""
    n = x.shape[-2]
    pad = -n % blk
    xp = torch.nn.functional.pad(x, (0, 0, 0, pad))
    sums = xp.reshape(*x.shape[:-2], -1, blk, x.shape[-1]).sum(-2)
    counts = torch.full((sums.shape[-2], 1), float(blk), dtype=x.dtype)
    counts[-1] = blk - pad
    return sums / counts


def _lightx2v_block_sets(q, k, topk_ratio, blkq=VSA_TILE_TOKENS, blkk=VSA_TILE_TOKENS):
    arg_k = k - k.mean(dim=-2, keepdim=True)
    score = _mean_pool(q, blkq) @ _mean_pool(arg_k, blkk).transpose(-1, -2)
    n_blocks = score.shape[-1]
    topk = max(1, min(n_blocks, int(topk_ratio * n_blocks)))
    return torch.topk(score, topk, dim=-1).indices, topk


def _ours_block_sets(q, k, geometry, sparsity, q_block_tiles=1):
    a = geometry.averaging_matrix(torch.float64)  # [n_tiles, padded_len]
    q_c = sla_query_averaging_matrix(geometry, q_block_tiles).to(torch.float64) @ geometry.pack_rows(q, dim=-2)
    k_c = a @ geometry.pack_rows(k, dim=-2)
    scores = q_c @ k_c.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores + torch.where(geometry.is_candidate, 0.0, -math.inf).to(scores.dtype)
    kk = compute_topk_sla(sparsity, int(geometry.is_candidate.sum()))
    ids = torch.topk(scores, kk, dim=-1).indices
    real = geometry.tile_ids >= 0
    return ids[..., real, :], kk


@pytest.mark.parametrize("q_block_tiles", [1, 2])
@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("sp_factor", [1, 32])
def test_sla_geometry(case, sp_factor, q_block_tiles):
    seq_len, grid = _CASES[case]
    g = build_sla_geometry(seq_len, grid, sp_factor=sp_factor, q_block_tiles=q_block_tiles)
    n_real = math.ceil(seq_len / VSA_TILE_TOKENS)
    assert g.tiles_per_shard % q_block_tiles == 0 and g.n_tiles - g.n_pad_tiles == n_real
    assert int(g.valid_counts.sum()) == seq_len
    assert g.valid_counts[n_real - 1] == seq_len - (n_real - 1) * VSA_TILE_TOKENS
    assert not g.is_exempt.any() and not g.is_3d.any()
    assert torch.equal(g.is_candidate, g.valid_counts > 0)
    assert torch.equal(g.tile_ids[:n_real], torch.arange(n_real))
    assert (g.tile_ids[n_real:] == -1).all()
    # raster order: the padded sequence is the packed one plus a pad tail
    assert torch.equal(g.gather_index[:seq_len], torch.arange(seq_len))
    assert (g.gather_index[seq_len:] == -1).all()
    x = torch.randn(seq_len, 3)
    assert torch.equal(g.unpack_rows(g.pack_rows(x)), x)
    assert torch.equal(g.permute_metadata(x)[:seq_len], x)
    # pad slots repeat a real row (rope/AdaLN tables stay finite)
    assert ((g.row_source >= 0) & (g.row_source < seq_len)).all()


@pytest.mark.parametrize("q_block_tiles", [1, 2])
@pytest.mark.parametrize("sparsity", [0.85, 0.9])
@pytest.mark.parametrize("sp_factor", [1, 32])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_sla_selection_matches_lightx2v(case, sp_factor, sparsity, q_block_tiles):
    seq_len, grid = _CASES[case]
    heads, head_dim = 2, 32
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(1, heads, seq_len, head_dim, generator=gen, dtype=torch.float64)
    # a shared offset in K makes smooth-K matter for the scores, not the per-row ranking
    k = torch.randn(1, heads, seq_len, head_dim, generator=gen, dtype=torch.float64) + 0.5
    g = build_sla_geometry(seq_len, grid, sp_factor=sp_factor, q_block_tiles=q_block_tiles)

    ref, ref_k = _lightx2v_block_sets(q, k, 1 - sparsity, blkq=q_block_tiles * VSA_TILE_TOKENS)
    ours, our_k = _ours_block_sets(q, k, g, sparsity, q_block_tiles)
    assert our_k == ref_k
    # every 64-row tile takes its query block's selection
    ref = ref.repeat_interleave(q_block_tiles, dim=-2)[..., : ours.shape[-2], :]
    assert torch.equal(ours.sort(dim=-1).values, ref.sort(dim=-1).values)


def test_compute_topk_sla_floors():
    assert compute_topk_sla(0.85, 100) == 15
    assert compute_topk_sla(0.85, 6) == 1  # int(0.9) = 0, clamped to 1
    assert compute_topk_sla(0.0, 7) == 7
