# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip validation of the qr-ring SHARD-LOCAL sparse attention mode
(ttnn.transformer.sparse_sdpa_stats_shard_local) — the keystone for the fused Q-gather ring.

Production keeps the KVPE latent cache SHARDED and stationary across SP (block-cyclic), and gathers the
queries instead of the O(T) KV. Each SP rank then attends its q against ONLY its local KV stripe, producing a
per-shard softmax partial (O, m, l); the partials flash-merge (online softmax) to the exact full-attention
output. This test proves, on ONE chip (no mesh/fabric), that:

  golden  = sparse_sdpa(q, kv_natural, indices_natural)                         # full top-k attention
  ==  merge_s [ sparse_sdpa_stats_shard_local(q, stripe_s, indices_natural, sp, chunk_local, s) ]

where stripe_s is shard s's block-cyclic physical stripe [1,1,T/sp,K_DIM]. The op remaps each natural top-k
index to its LOCAL page (or drops it if it lands in another stripe), and emits the identity partial
(O=0, m=-BIG, l=0) for a query that selected NO keys from stripe s — so empty stripes contribute nothing.

Coverage is deliberate: rows that hit all shards, rows with a sentinel tail (nv < TOPK), and rows whose top-k
all land in ONE shard (forcing empty stripes elsewhere → the identity-partial path).
"""
import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import MASKED_INDEX, golden, pcc, to_dev

K_DIM = 576
V_DIM = 512


def _shard_of(n, chunk_local, sp):
    return (n // chunk_local) % sp


def _stripe_natural_ids(shard, chunk_local, sp, T):
    """Ordered natural token ids that stripe `shard` physically holds (block-cyclic). Local page p maps to
    natural token (p//chunk_local)*chunk_local*sp + shard*chunk_local + p%chunk_local."""
    num_slabs = T // (chunk_local * sp)
    slab = torch.arange(num_slabs).view(-1, 1)
    r = torch.arange(chunk_local).view(1, -1)
    return (slab * (chunk_local * sp) + shard * chunk_local + r).reshape(-1)  # [T/sp]


def _build_indices(S, T, TOPK, sp, chunk_local, gen):
    """Natural top-k indices [1,1,S,TOPK] (valid prefix + sentinel tail). Rows are a mix:
    - full rows (nv==TOPK, spread across T -> keys in every shard)
    - short rows (nv < TOPK -> a sentinel tail)
    - single-shard rows (all nv keys inside ONE shard's tokens -> the OTHER sp-1 stripes are empty)"""
    idx = torch.full((1, 1, S, TOPK), MASKED_INDEX, dtype=torch.int64)
    shard_tokens = [_stripe_natural_ids(s, chunk_local, sp, T) for s in range(sp)]
    for s in range(S):
        kind = s % 3
        if kind == 0:  # full, spread
            nv = TOPK
            perm = torch.randperm(T, generator=gen)[:nv]
        elif kind == 1:  # short tail
            nv = max(1, TOPK // 2 - (s % 5))
            perm = torch.randperm(T, generator=gen)[:nv]
        else:  # all keys in a single shard -> empties elsewhere
            tgt = shard_tokens[s % sp]
            nv = min(TOPK, tgt.numel())
            perm = tgt[torch.randperm(tgt.numel(), generator=gen)[:nv]]
        idx[0, 0, s, :nv] = perm
    return idx


def _merge(partials, scale):
    """Host online-softmax merge of [(O_norm, m_raw, l)] -> normalized out (stats are col-0 of [.,.,.,32])."""
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
@pytest.mark.parametrize("sp", [2, 4], ids=lambda s: f"sp{s}")
@pytest.mark.parametrize(
    "H,S,T,TOPK,chunk_local",
    [(32, 96, 512, 128, 32), (32, 96, 1024, 128, 64), (64, 96, 4096, 1024, 512)],
    ids=["t512cl32", "t1024cl64", "h64t4096cl512"],
)
def test_sparse_sdpa_shard_local(device, sp, H, S, T, TOPK, chunk_local):
    assert T % (chunk_local * sp) == 0
    scale = K_DIM**-0.5
    gen = torch.Generator().manual_seed(1000 * sp + T + chunk_local)
    q = torch.randn(1, H, S, K_DIM, generator=gen, dtype=torch.float32)
    kv = torch.randn(1, 1, T, K_DIM, generator=gen, dtype=torch.float32)
    indices = _build_indices(S, T, TOPK, sp, chunk_local, gen)

    ref = golden(q, kv, indices, scale, V_DIM)  # [1,H,S,V] full top-k attention

    tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)

    partials = []
    for s in range(sp):
        stripe = kv[:, :, _stripe_natural_ids(s, chunk_local, sp, T), :]  # [1,1,T/sp,K_DIM]
        tt_stripe = to_dev(stripe.to(torch.bfloat16), device, ttnn.bfloat16)
        tt_shard = to_dev(torch.tensor([[[[s]]]], dtype=torch.int32), device, ttnn.uint32)  # this stripe's id
        outs = ttnn.transformer.sparse_sdpa_stats_shard_local(
            tt_q, tt_stripe, tt_idx, tt_shard, V_DIM, sp=sp, chunk_local=chunk_local, scale=scale, k_chunk_size=32
        )
        assert len(outs) == 3, f"expected [O,m,l], got {len(outs)}"
        O = ttnn.to_torch(outs[0]).float()
        m = ttnn.to_torch(outs[1]).float()[..., 0:1]
        l = ttnn.to_torch(outs[2]).float().sum(dim=-1, keepdim=True)
        partials.append((O, m, l))
        ttnn.deallocate(tt_stripe)

    out = _merge(partials, scale)
    p = pcc(out, ref)
    assert p >= 0.99, f"shard-local merge PCC {p:.5f} (sp={sp}, T={T}, cl={chunk_local}) < 0.99"
