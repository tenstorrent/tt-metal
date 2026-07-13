# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device online-softmax MERGE of per-shard sparse partials (ttnn eltwise, no CCL).

The qr-ring cross-SP merge, done with device ops instead of on the host: given each stripe's normalized output
O and softmax stats (m raw row-max in col 0, l denom spread across 32 cols), combine to the exact full
attention. This validates the merge MATH on device (single chip, no fabric) — the reusable core the multi-device
mla wiring wraps with all_gather(m)+reduce_scatter(w*O,w). Partials come from sparse_sdpa_stats_shard_local.

    M = max_s (scale*m_s);  w_s = exp(scale*m_s - M) * sum(l_s);  out = sum_s w_s O_s / sum_s w_s
"""
import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import MASKED_INDEX, golden, pcc, to_dev

K_DIM = 576
V_DIM = 512


def _stripe_natural_ids(shard, chunk_local, sp, T):
    num_slabs = T // (chunk_local * sp)
    slab = torch.arange(num_slabs).view(-1, 1)
    r = torch.arange(chunk_local).view(1, -1)
    return (slab * (chunk_local * sp) + shard * chunk_local + r).reshape(-1)


def _build_indices(S, T, TOPK, sp, chunk_local, gen):
    idx = torch.full((1, 1, S, TOPK), MASKED_INDEX, dtype=torch.int64)
    stoks = [_stripe_natural_ids(s, chunk_local, sp, T) for s in range(sp)]
    for s in range(S):
        if s % 3 == 2:  # single-stripe row -> empties elsewhere (identity partial)
            tgt = stoks[s % sp]
            perm = tgt[torch.randperm(tgt.numel(), generator=gen)[: min(TOPK, tgt.numel())]]
        else:
            nv = TOPK if s % 3 == 0 else max(1, TOPK // 2)
            perm = torch.randperm(T, generator=gen)[:nv]
        idx[0, 0, s, : perm.numel()] = perm
    return idx


def _merge_on_device(partials, scale):
    """partials: list of (O,m,l) ttnn ROW_MAJOR tensors. Returns merged O (ttnn TILE [1,H,S,V])."""
    tl = ttnn.TILE_LAYOUT
    # scale*m col0 per shard, and the running max M across shards.
    scaled_m, l_sum, O_tile = [], [], []
    M = None
    for O, m, l in partials:
        mt = ttnn.to_layout(m, tl)
        sm = ttnn.multiply(mt, scale)  # [1,H,S,32], value in col 0
        sm0 = ttnn.slice(sm, [0, 0, 0, 0], [sm.shape[0], sm.shape[1], sm.shape[2], 1])  # [1,H,S,1]
        scaled_m.append(sm0)
        lt = ttnn.to_layout(l, tl)
        l_sum.append(ttnn.sum(lt, dim=-1, keepdim=True))  # [1,H,S,1]
        O_tile.append(ttnn.to_layout(O, tl))
        M = sm0 if M is None else ttnn.maximum(M, sm0)
    num, den = None, None
    for sm0, ls, Ot in zip(scaled_m, l_sum, O_tile):
        w = ttnn.multiply(ttnn.exp(ttnn.subtract(sm0, M)), ls)  # [1,H,S,1]
        term = ttnn.multiply(Ot, w)  # broadcast w over V -> [1,H,S,V]
        num = term if num is None else ttnn.add(num, term)
        den = w if den is None else ttnn.add(den, w)
    return ttnn.multiply(num, ttnn.reciprocal(den))  # [1,H,S,V]


@run_for_blackhole()
@pytest.mark.parametrize("sp", [2, 4], ids=lambda s: f"sp{s}")
@pytest.mark.parametrize("H,S,T,TOPK,chunk_local", [(32, 96, 1024, 128, 64)], ids=["h32s96t1024k128cl64"])
def test_merge_ondevice(device, sp, H, S, T, TOPK, chunk_local):
    assert T % (chunk_local * sp) == 0
    scale = K_DIM**-0.5
    gen = torch.Generator().manual_seed(7 * sp + T)
    q = torch.randn(1, H, S, K_DIM, generator=gen)
    kv = torch.randn(1, 1, T, K_DIM, generator=gen)
    indices = _build_indices(S, T, TOPK, sp, chunk_local, gen)
    ref = golden(q, kv, indices, scale, V_DIM)

    tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)

    partials = []
    for s in range(sp):
        stripe = kv[:, :, _stripe_natural_ids(s, chunk_local, sp, T), :]
        tt_stripe = to_dev(stripe.to(torch.bfloat16), device, ttnn.bfloat16)
        tt_shard = to_dev(torch.tensor([[[[s]]]], dtype=torch.int32), device, ttnn.uint32)
        outs = ttnn.transformer.sparse_sdpa_stats_shard_local(
            tt_q, tt_stripe, tt_idx, tt_shard, V_DIM, sp=sp, chunk_local=chunk_local, scale=scale, k_chunk_size=32
        )
        partials.append((outs[0], outs[1], outs[2]))

    out = _merge_on_device(partials, scale)
    p = pcc(ttnn.to_torch(out).float(), ref)
    assert p >= 0.99, f"on-device merge PCC {p:.5f} (sp={sp}) < 0.99"
