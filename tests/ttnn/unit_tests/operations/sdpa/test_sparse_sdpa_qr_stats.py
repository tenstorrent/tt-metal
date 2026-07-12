# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip validation of the qr-ring stat export (ttnn.transformer.sparse_sdpa_stats).

Proves the device op emits correct per-shard softmax stats (m, l) alongside the normalized output O, and
that flash-merging per-shard partials on the host reproduces the full-T golden — all on ONE chip, no fabric.
This is the keystone: the (O, m, l) contract everything downstream (the cross-SP ring) is checked against.

Split the top-k slots into `sp` disjoint groups (each non-empty per row). Run sparse_sdpa_stats once per
group over the SAME full KV with the other groups masked -> that group's partial (O_i, m_i, l_i). Merge:
    M = max_i (scale*m_i); w_i = exp(scale*m_i - M) * l_i; out = sum_i w_i O_i / sum_i w_i.
"""
import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import MASKED_INDEX, golden, make_inputs, pcc, to_dev

K_DIM = 576
V_DIM = 512


def _shard_indices(indices, sp):
    """Split the TOPK slots into `sp` disjoint groups; return per-shard index tensors with each shard's g
    valid indices COMPACTED to the front (slots [0,g)) and the rest MASKED as a contiguous tail — the op
    requires sentinels to be a contiguous tail. (The real multi-device wiring must likewise compact each
    shard's local indices to the front.)"""
    topk = indices.shape[-1]
    assert topk % sp == 0
    g = topk // sp
    out = []
    for s in range(sp):
        m = torch.full_like(indices, MASKED_INDEX)
        m[..., :g] = indices[..., s * g : (s + 1) * g]  # valid prefix, masked tail
        out.append(m)
    return out


def _merge(partials, scale):
    """Host online-softmax merge of [(O_norm, m_raw, l)] -> normalized out. Stats are col-0 of [.,.,.,32]."""
    ms = torch.stack([scale * p[1] for p in partials])
    M = ms.max(dim=0).values
    num, den = None, 0.0
    for O_norm, m_raw, l in partials:
        w = torch.exp(scale * m_raw - M) * l
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        term = w * O_norm
        num = term if num is None else num + term
        den = den + w
    return num / den.clamp_min(1e-30)


@run_for_blackhole()
@pytest.mark.parametrize("sp", [2, 4, 8], ids=lambda s: f"sp{s}")
@pytest.mark.parametrize("H,S,T,TOPK", [(32, 64, 512, 64)], ids=["h32s64t512k64"])
def test_sparse_sdpa_stats_single_chip(device, sp, H, S, T, TOPK):
    scale = K_DIM**-0.5
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 10**9, seed=sp)  # all-valid
    ref = golden(q, kv, indices, scale, V_DIM)  # full-T normalized golden [1,H,S,V]

    tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_kv = to_dev(kv.to(torch.bfloat16), device, ttnn.bfloat16)

    partials = []
    o_from_stats, o_plain = None, None
    for sh, idx in enumerate(_shard_indices(indices, sp)):
        tt_idx = to_dev(idx.to(torch.int32), device, ttnn.uint32)
        outs = ttnn.transformer.sparse_sdpa_stats(tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=32)
        assert len(outs) == 3, f"expected [O,m,l], got {len(outs)}"
        O = ttnn.to_torch(outs[0]).float()  # [1,H,S,V]
        # m: raw row-max is finalized in col 0. l: the softmax denom is spread across the 32 tile columns
        # (the op's per-column partial sums, which normalize() finalizes via a col-identity matmul) -> sum them.
        m = ttnn.to_torch(outs[1]).float()[..., 0:1]  # [1,H,S,1]
        l = ttnn.to_torch(outs[2]).float().sum(dim=-1, keepdim=True)  # [1,H,S,1]
        partials.append((O, m, l))
        # sanity: O from the stats op == plain sparse_sdpa on the same (masked) indices
        O_plain = ttnn.to_torch(ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=32))
        p_o = pcc(O, O_plain.float())
        assert p_o >= 0.999, f"stats-op O diverges from plain op (shard {sh}): PCC {p_o:.5f}"

    out = _merge(partials, scale)
    p = pcc(out, ref)
    assert p >= 0.99, f"qr merge PCC {p:.5f} (sp={sp}) < 0.99"
