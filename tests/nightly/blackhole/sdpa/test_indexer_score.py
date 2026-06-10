# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for the DeepSeek-V3.2 DSA ``indexer_score`` op
(design: models/demos/deepseek_v32/INDEXER_OP.md).

    score[b, s, t] = sum_h relu(q[b, h, s, :] . k[b, t, :]) * w[b, h, s]

Causality from scalar ``chunk_start``: key ``t`` visible to query ``s`` iff
``t <= chunk_start + s``; future columns are -inf.

Main case: Galaxy chunked prefill, 5K queries vs 55K keys, SP=8 (640
queries/device).  SP enters only via ``chunk_start``, so this is single-chip
with ``sp_rank`` selecting the ring position.  The op under test currently
runs the torch reference; swap in the ttnn op where marked.
"""

import pytest
import torch

import ttnn

GLX_HEADS, GLX_DIM = 64, 128
GLX_SQ = 640  # queries per device (5120 chunk / SP=8)
GLX_T = 56320  # all-gathered keys: 50K history + 5K chunk = 55K, tile-aligned
GLX_HISTORY = GLX_T - 8 * GLX_SQ  # 51200 keys visible to every query


def indexer_score_ref(q, k, w, chunk_start):
    """Per-head fp32 accumulation (a full [Hi,Sq,T] tensor is ~9 GB here)."""
    b, hi, sq, _ = q.shape
    t = k.shape[2]
    q, k, w = q.float(), k.float(), w.float()
    score = torch.zeros(b, sq, t)
    for h in range(hi):
        score += torch.relu(q[:, h] @ k[:, 0].transpose(-2, -1)) * w[:, h]
    future = torch.arange(t).unsqueeze(0) > chunk_start + torch.arange(sq).unsqueeze(1)
    return score.masked_fill(future, float("-inf")).unsqueeze(1)


def indexer_score(q, k, w, chunk_start, device):
    """Op under test (skeleton: compiles/dispatches, output garbage -> bad PCC)."""

    def dev(t):
        return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    out = ttnn.experimental.deepseek.indexer_score(dev(q), dev(k), dev(w), chunk_start_idx=chunk_start)
    return ttnn.to_torch(out)


@pytest.mark.parametrize("sp_rank", [0, 7])
def test_indexer_score_glx_chunked(device, sp_rank):
    chunk_start = GLX_HISTORY + sp_rank * GLX_SQ
    g = torch.Generator().manual_seed(42)
    q = torch.randn(1, GLX_HEADS, GLX_SQ, GLX_DIM, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, GLX_T, GLX_DIM, generator=g, dtype=torch.bfloat16)
    # negative gates make real scores negative — zero-filled columns would win topk
    w = torch.randn(1, GLX_HEADS, GLX_SQ, 1, generator=g, dtype=torch.bfloat16)

    out = indexer_score(q, k, w, chunk_start, device)
    ref = indexer_score_ref(q, k, w, chunk_start)

    assert out.shape == (1, 1, GLX_SQ, GLX_T)
    # -inf maps must agree exactly (<= bf16 lowest counts as masked on device)
    masked = ref == float("-inf")
    assert torch.equal(out <= torch.finfo(torch.bfloat16).min, masked)
    # visible values by PCC (0.999 floor for the bf16 device op)
    a, b = out[~masked].flatten().float(), ref[~masked].flatten().float()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pcc >= 0.999, f"PCC {pcc} < 0.999"
    assert (ref[~masked] < 0).any()
