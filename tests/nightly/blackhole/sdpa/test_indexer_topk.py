# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Two-part DSA indexing pipeline test: ``indexer_score`` -> ``topk_xl``.

DeepSeek-V3.2 sparse attention selects, per query, the **top-2048** keys by the
lightning-indexer score and feeds those key indices to the sparse MLA
(INDEXER_OP.md: "After the score, top-k(2048) indices feed the sparse MLA"; the
score is emitted ROW_MAJOR bf16 "designed for a row-major top-k written
alongside"). ``topk_xl`` is that row-major top-k.

This test runs the two ops back to back on device. The indexer score output is
fed directly into ``topk_xl`` with **no layout or shape massaging** -- the score
is already ROW_MAJOR bf16, and topk_xl flattens leading dims, so the
``[1,1,Sq,T]`` score becomes ``num_rows=Sq, n=T`` and the result is
``[1,1,Sq,K]`` uint32 key indices. It mirrors ``test_indexer_score.py`` (same
reference, same GLX production shape / knobs) and adds the selection stage.

What it checks (the contract the sparse-MLA consumer relies on):
  1. Part 1 -- device scores match the fp32 reference (PCC + exact -inf map).
  2. Mask safety -- no selected index is a future / -inf column. Random (some
     negative) indexer weights make real scores negative, so a masked column
     cannot hide as a low-but-valid score and be picked.
  3. Selection quality -- topk_xl run on the device scores selects essentially
     the same values torch.topk would on those same scores: the count of picks
     strictly below the true K-th value is negligible (bf16 tie-boundary only).
  4. End-to-end quality -- the device pipeline's selected key set overlaps the
     fp32-reference top-k key set above a high floor.

Note on bf16 ties: topk_xl is not bit-for-bit torch-equivalent under exact bf16
ties (its own -inf tie test is skipped for the same reason). When the visible
key count is barely above K (so the K-th cut sits in a dense tie cluster) it can
emit a duplicate index, dropping one boundary key. This is inherent to topk_xl
(reproduces identically on a plain 2D tensor, independent of the indexer) and is
boundary-only, so it is bounded here rather than asserted away.
"""

import pytest
import torch

import ttnn

# --- GLX chunked-prefill production shape (matches test_indexer_score.py) ---
GLX_HEADS, GLX_DIM = 64, 128
GLX_SQ = 640  # queries per device (5120 chunk / SP=8)
GLX_T = 56320  # all-gathered keys: 50K history + 5K chunk, tile-aligned
GLX_HISTORY = GLX_T - 8 * GLX_SQ  # 51200 keys visible to every query
DSA_TOPK = 2048  # DeepSeek-V3.2 selects top-2048 keys per query

BF16_MIN = torch.finfo(torch.bfloat16).min

# Small single-chip shape for fast mechanics / masking / tie coverage.
MINI = dict(heads=64, dim=128, sq=64, t=256)


def indexer_score_ref(q, k, w, chunk_start):
    """Per-head fp32 accumulation; returns [1,1,Sq,T] with future cols = -inf."""
    b, hi, sq, _ = q.shape
    t = k.shape[2]
    q, k, w = q.float(), k.float(), w.float()
    score = torch.zeros(b, sq, t)
    for h in range(hi):
        score += torch.relu(q[:, h] @ k[:, 0].transpose(-2, -1)) * w[:, h]
    future = torch.arange(t).unsqueeze(0) > chunk_start + torch.arange(sq).unsqueeze(1)
    return score.masked_fill(future, float("-inf")).unsqueeze(1)


def make_inputs(heads, dim, sq, t, seed=42):
    """q [1,Hi,Sq,D], k [1,1,T,D], weights [1,Hi,Sq,1], all bf16.

    Random weights => some gates negative => some real scores negative, so a
    masked (-inf) column can never masquerade as a valid low score.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, heads, sq, dim, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, t, dim, generator=g, dtype=torch.bfloat16)
    w = torch.randn(1, heads, sq, 1, generator=g, dtype=torch.bfloat16)
    return q, k, w


def _to_dev(t, device):
    return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)


def run_pipeline(q, k, w, chunk_start, topk, device, score_cfg):
    """Run indexer_score -> topk_xl on device. Returns (scores[Sq,T] bf16->float,
    indices[Sq,topk] int64), as host torch tensors."""
    score = ttnn.experimental.deepseek.indexer_score(
        _to_dev(q, device),
        _to_dev(k, device),
        _to_dev(w, device),
        chunk_start_idx=chunk_start,
        **({} if score_cfg is None else {"program_config": score_cfg}),
    )
    # The score is already ROW_MAJOR bf16; topk_xl flattens leading dims, so the
    # [1,1,Sq,T] score feeds in directly (num_rows=Sq, n=T) with no conversion.
    assert score.layout == ttnn.ROW_MAJOR_LAYOUT, f"indexer score layout {score.layout} != ROW_MAJOR"
    assert score.dtype == ttnn.bfloat16
    idx = ttnn.experimental.topk_xl(score, k=topk)

    sq, t = q.shape[2], k.shape[2]
    score_t = ttnn.to_torch(score).reshape(sq, t).float()
    idx_t = ttnn.to_torch(idx, dtype=torch.uint32).to(torch.int64).reshape(sq, topk)
    return score_t, idx_t


def _check(q, k, w, chunk_start, topk, device, score_cfg, pcc_floor=0.999, overlap_floor=0.95):
    sq, t = q.shape[2], k.shape[2]
    s2d, idx = run_pipeline(q, k, w, chunk_start, topk, device, score_cfg)
    ref = indexer_score_ref(q, k, w, chunk_start)[0, 0]  # [Sq, T] fp32, future=-inf

    # ---- 1. Part-1 scores correct: exact -inf map + PCC on visible values ----
    masked = ref == float("-inf")
    assert torch.equal(s2d <= BF16_MIN, masked), "device -inf map != reference"
    a, b = s2d[~masked].flatten(), ref[~masked].flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pcc >= pcc_floor, f"indexer score PCC {pcc} < {pcc_floor}"
    assert (ref[~masked] < 0).any(), "expected some negative valid scores (neg weights)"

    sel = torch.gather(s2d, 1, idx)  # device scores at selected keys

    # ---- 2. Mask safety: no future / -inf column selected ----
    assert idx.min() >= 0 and idx.max() < t, "selected index out of range"
    last_visible = (chunk_start + torch.arange(sq)).clamp(max=t - 1)  # max valid t per row
    assert torch.all(idx <= last_visible.unsqueeze(1)), "selected a future (masked) key"
    assert torch.all(sel > BF16_MIN), "selected an -inf column"

    # ---- 3. Selection quality vs torch.topk on the SAME device scores ----
    # topk_xl must not reach below the true K-th value (except bf16 tie-boundary
    # slips, which are vanishingly rare). This is the "compatible" core check.
    kth = torch.topk(s2d, topk, dim=-1, largest=True).values[:, -1:]
    below_cut = int((sel < kth).sum())
    slots = sq * topk
    assert below_cut <= max(8, slots // 1000), f"{below_cut}/{slots} picks below true K-th value"

    # ---- 4. End-to-end quality vs fp32 reference selection ----
    _, ref_idx = torch.topk(ref, topk, dim=-1, largest=True, sorted=True)
    overlap = sum(len(set(idx[r].tolist()) & set(ref_idx[r].tolist())) for r in range(sq)) / float(slots)
    assert overlap >= overlap_floor, f"device/fp32 top-k key overlap {overlap:.4f} < {overlap_floor}"

    dup_rows = sum(1 for r in range(sq) if idx[r].unique().numel() < topk)
    print(
        f"[indexer->topk] sq={sq} t={t} k={topk}: pcc={pcc:.5f} below_cut={below_cut}/{slots} "
        f"dup_rows={dup_rows}/{sq} fp32_overlap={overlap:.5f}"
    )
    return pcc, overlap


@pytest.mark.parametrize("chunk_start", [128, 192], ids=["mini_rank0", "mini_rank1"])
def test_indexer_topk_mini(device, chunk_start):
    """Fast mechanics: indexer->topk chaining, tie handling, masking safety.

    K=128 with min visible keys = chunk_start+1 (>=129): the cut sits in a dense
    tie cluster, so this also exercises topk_xl's boundary tie behavior.
    """
    q, k, w = make_inputs(**MINI)
    _check(q, k, w, chunk_start, topk=128, device=device, score_cfg=None)


def test_indexer_topk_multicore(device):
    """Larger shape so query rows split across many cores; K=512."""
    heads, dim, sq, t, chunk_start = 64, 128, 128, 2048, 512
    q, k, w = make_inputs(heads, dim, sq, t)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=32, head_group_size=0)
    _check(q, k, w, chunk_start, topk=512, device=device, score_cfg=cfg)


@pytest.mark.parametrize("sp_rank", [0, 7], ids=["rank0", "rank7"])
def test_indexer_topk_production(device, sp_rank):
    """GLX chunked prefill production case: full indexer score -> top-2048 select.

    history (51200) >> K (2048): every query has far more valid keys than K, so
    there are no forced -inf picks and the selection is duplicate-free.
    """
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=512, head_group_size=0)
    chunk_start = GLX_HISTORY + sp_rank * GLX_SQ
    q, k, w = make_inputs(GLX_HEADS, GLX_DIM, GLX_SQ, GLX_T)
    _check(q, k, w, chunk_start, topk=DSA_TOPK, device=device, score_cfg=cfg, overlap_floor=0.98)
