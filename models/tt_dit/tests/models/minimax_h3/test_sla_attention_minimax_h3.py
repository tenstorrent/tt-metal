# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""SLA mode end to end on the host: MiniMaxH3VSACoarseStage's device-path math against LightX2V's SLA attention.

CPU-only. Our side replays what the device runs per SP shard: the stage's own sharded averaging matrices
(padded pooling), the gathered pooled K, the candidate mask, top-k over the padded per-shard numbering, the
kernel's padded -> global block mapping (coarse_slots_shift / coarse_real_per_shard), and block-sparse softmax
over each selected block's valid rows (block_counts). The oracle is LightX2V dynamic_sparse_attn: get_block_map
(smooth-K, mean pooling, floor top-k) and block-sparse attention over the packed sequence.
"""

import math

import pytest
import torch

from ....models.transformers.minimax_h3 import vsa_stages_minimax_h3
from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, build_sla_geometry

# text 512 + cond video 2 x 1008 + audio 400 + 31 x 24 x 42 video rows: 768p 10 s fl2va
_SEQ = {"tiny": 1000, "768p10s": 512 + 2016 + 400 + 31 * 24 * 42}


def _attend(q, k, v, keys):
    p = torch.softmax(q @ k[keys].T / math.sqrt(q.shape[-1]), dim=-1)
    return p @ v[keys]


def _lightx2v_sla(q, k, v, topk_ratio, blkq, blkk=VSA_TILE_TOKENS):
    heads, seq_len, d = q.shape

    def mean_pool(x, blk):
        pad = -seq_len % blk
        sums = torch.nn.functional.pad(x, (0, 0, 0, pad)).reshape(heads, -1, blk, d).sum(-2)
        counts = torch.full((sums.shape[1], 1), float(blk), dtype=x.dtype)
        counts[-1] = blk - pad
        return sums / counts

    scores = mean_pool(q, blkq) @ mean_pool(k - k.mean(-2, keepdim=True), blkk).transpose(-1, -2)
    n_blocks = scores.shape[-1]
    ids = torch.topk(scores, max(1, min(n_blocks, int(topk_ratio * n_blocks))), dim=-1).indices
    out = torch.empty_like(q)
    for h in range(heads):
        for qb in range(scores.shape[1]):
            keys = torch.cat([torch.arange(b * blkk, min(seq_len, (b + 1) * blkk)) for b in ids[h, qb].tolist()])
            rows = slice(qb * blkq, min(seq_len, (qb + 1) * blkq))
            out[h, rows] = _attend(q[h, rows], k[h], v[h], keys)
    return out, ids


def _ours_sla(q, k, v, geometry, sparsity, q_block_tiles, monkeypatch):
    monkeypatch.setattr(vsa_stages_minimax_h3, "from_torch", lambda x, **_: x)
    heads, _, d = q.shape
    stage = vsa_stages_minimax_h3.MiniMaxH3VSACoarseStage(
        geometry,
        sparsity=sparsity,
        selection="sla",
        sla_q_block_tiles=q_block_tiles,
        padded_pooling=True,
        head_dim=d,
        mesh_device=None,
        sp_axis=1,
    )
    qp, kp, vp = (geometry.pack_rows(x, dim=-2) for x in (q, k, v))
    sp, tiles, slots = geometry.sp_factor, geometry.tiles_per_shard, stage.slots_per_shard
    rows = geometry.padded_len // sp
    a_q = stage.a_t_q.reshape(geometry.padded_len, slots).to(q.dtype)
    a_kv = stage.a_t_kv.reshape(geometry.padded_len, slots).to(q.dtype)
    shard = lambda x, s: x[..., s * rows : (s + 1) * rows, :]
    k_c_t = torch.cat([shard(kp, s).transpose(-1, -2) @ shard(a_kv, s) for s in range(sp)], dim=-1)
    cand_mask = stage.cand_mask.reshape(-1).to(q.dtype)
    counts = stage.block_counts_tensor().reshape(-1)
    pad_slots = slots - tiles
    out = torch.zeros_like(qp)
    selected = {}
    for s in range(sp):
        q_c = (shard(qp, s).transpose(-1, -2) @ shard(a_q, s)).transpose(-1, -2)
        ids = torch.topk(q_c @ k_c_t + cand_mask, stage.k_pad, dim=-1).indices[:, :tiles, : stage.k]
        for h in range(heads):
            for r in range(tiles):
                t = s * tiles + r
                blocks = sorted({b - (b >> stage.coarse_slots_shift) * pad_slots for b in ids[h, r].tolist()})
                selected[h, t] = blocks
                keys = torch.cat(
                    [torch.arange(b * VSA_TILE_TOKENS, b * VSA_TILE_TOKENS + int(counts[b])) for b in blocks]
                )
                q_rows = slice(t * VSA_TILE_TOKENS, (t + 1) * VSA_TILE_TOKENS)
                out[h, q_rows] = _attend(qp[h, q_rows], kp[h], vp[h], keys)
    return geometry.unpack_rows(out, dim=-2), selected


@pytest.mark.parametrize("q_block_tiles", [1, 2])
@pytest.mark.parametrize("sp_factor", [1, 8])
@pytest.mark.parametrize("case", sorted(_SEQ))
def test_sla_attention_matches_lightx2v(case, sp_factor, q_block_tiles, monkeypatch):
    seq_len, heads, d, sparsity = _SEQ[case], 2, 32, 0.85
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(heads, seq_len, d, generator=gen, dtype=torch.float64) for _ in range(3))
    k = k + 0.5  # smooth-K shifts every score row by a constant: the selection must not change
    g = build_sla_geometry(seq_len, (1, 1, 1), sp_factor=sp_factor, q_block_tiles=q_block_tiles)

    ref, ref_ids = _lightx2v_sla(q, k, v, 1 - sparsity, q_block_tiles * VSA_TILE_TOKENS)
    ours, selected = _ours_sla(q, k, v, g, sparsity, q_block_tiles, monkeypatch)
    for (h, t), blocks in selected.items():
        if t * VSA_TILE_TOKENS < seq_len:
            assert blocks == sorted(ref_ids[h, t // q_block_tiles].tolist()), (h, t)
    torch.testing.assert_close(ours, ref, rtol=0, atol=1e-12)
