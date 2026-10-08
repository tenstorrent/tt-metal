# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa_msa packed token groups (16 query heads per KV group).

The packed kernels put two tokens in each 32-row Q tile row and gather the union of a group's selected blocks
once (per-token -inf masks hide a block from tokens that did not select it). These cases stress what the
grouping has to get right: realistic block locality (groups actually form), token groups that straddle a
128-token block boundary (two diagonal blocks in one group), ragged per-core token counts, groups cut at a
KV-group boundary (GQA with odd S), sentinel tails, duplicate ids, and selections with no common block (the
group must fall back to fewer tokens). Each case is checked against the golden and against the legacy
one-token kernels (TT_MSA_PACKED_GROUP=0) on identical inputs.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_msa_test_utils import (
    BLK_KV,
    SENTINEL,
    pcc,
    run_op_msa_native,
    sparse_attention_ref_msa,
)

_D = 128
DEVICE_PCC = 0.99  # device vs fp32 golden (same floor as the op's other tests)
LEGACY_PCC = 0.9999  # packed vs legacy kernels: only the block order of the online softmax differs


def _local_indices(n_kv, S, T, topk, chunk_start, causal, gen, *, dup_rate=0.0, tail_rate=0.0, scatter=False):
    """Block ids with realistic locality: a few shared 'hot' blocks per KV group, the query's own block, and
    recent blocks; optional duplicates (of a past block) and sentinel tails. scatter=True draws independent
    random rows (neighbouring tokens may share no block)."""
    nblk = T // BLK_KV
    idx = torch.full((1, n_kv, S, topk), SENTINEL, dtype=torch.int32)
    for g in range(n_kv):
        hot = torch.randperm(nblk, generator=gen)[:3].tolist()
        for s in range(S):
            p = chunk_start + s
            limit = (p // BLK_KV + 1) if causal else nblk
            diag = p // BLK_KV if causal else None
            if scatter:
                cand = torch.randperm(limit, generator=gen)[:topk].tolist()
            else:
                cand = ([diag] if causal else []) + [h for h in hot if h < limit]
                cand += [max(0, limit - 1 - j) for j in range(1, 4)]
                cand += torch.randperm(limit, generator=gen)[:topk].tolist()
            row = []
            for b in cand:
                if b not in row and len(row) < topk:
                    row.append(b)
            if causal and diag not in row:
                row[0] = diag
            n = len(row)
            if tail_rate and torch.rand(1, generator=gen).item() < tail_rate:
                n = max(1, int(torch.randint(1, n + 1, (1,), generator=gen).item()))
                if causal and diag not in row[:n]:
                    row[0] = diag
            row = row[:n]
            if dup_rate and n > 2 and torch.rand(1, generator=gen).item() < dup_rate:
                past = [b for b in row if b != diag]
                row[-1] = past[0]  # a past block listed twice
            idx[0, g, s, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return idx


def _run(device, monkeypatch, group, q, k, v, idx, chunk_start, kv_dtype):
    monkeypatch.setenv("TT_MSA_PACKED_GROUP", str(group))
    return run_op_msa_native(q, k, v, idx, device, kv_dtype=kv_dtype, chunk_start_idx=chunk_start).float()


@run_for_blackhole()
@pytest.mark.parametrize("group", [2, 8])
@pytest.mark.parametrize(
    "n_kv,S,chunk_start,causal",
    [
        (1, 300, 1000, True),  # ragged per-core counts; groups straddle block boundaries (1000 % 128 != 0)
        (1, 37, 4064, True),  # one group straddles 4096 (two diagonal blocks)
        (2, 37, 640, True),  # GQA, odd S: a group is cut at the KV-group boundary
        (1, 300, None, False),  # non-causal: no diagonal masks, only per-token hidden blocks
    ],
    ids=["causal_s300", "causal_straddle", "gqa_odd_s", "noncausal"],
)
def test_msa_packed_matches_golden_and_legacy(device, monkeypatch, group, n_kv, S, chunk_start, causal):
    T = 64 * BLK_KV
    topk = 16
    gen = torch.Generator().manual_seed(S + n_kv + (chunk_start or 0))
    H = 16 * n_kv
    q = torch.randn(1, H, S, _D, generator=gen)
    k = torch.randn(1, n_kv, T, _D, generator=gen)
    v = torch.randn(1, n_kv, T, _D, generator=gen)
    idx = _local_indices(n_kv, S, T, topk, chunk_start or 0, causal, gen, tail_rate=0.2)
    gold = sparse_attention_ref_msa(q, k, v, idx, _D**-0.5, causal=causal, chunk_start_idx=chunk_start or 0)
    packed = _run(device, monkeypatch, group, q, k, v, idx, chunk_start, ttnn.bfloat8_b)
    legacy = _run(device, monkeypatch, 0, q, k, v, idx, chunk_start, ttnn.bfloat8_b)
    assert not torch.isnan(packed).any()
    p_gold, p_leg = pcc(packed, gold), pcc(packed, legacy)
    assert p_gold > DEVICE_PCC, f"packed vs golden pcc={p_gold}"
    assert p_leg > LEGACY_PCC, f"packed vs legacy pcc={p_leg}"
    # per-token floor: a wrong mask on one token would hide under the global PCC
    for s in range(S):
        ps = pcc(packed[:, :, s], legacy[:, :, s])
        assert ps > 0.999, f"token {s}: packed vs legacy pcc={ps}"


@run_for_blackhole()
@pytest.mark.parametrize("group", [4, 8])
def test_msa_packed_no_common_block_and_duplicates(device, monkeypatch, group):
    """Scattered selections (groups must shrink to keep a common lead block) with duplicate ids (attended
    twice, as the legacy kernel does) and sentinel tails; bf16 K/V. Compared with the legacy kernels."""
    S, T, topk, chunk_start = 64, 32 * BLK_KV, 16, None  # non-causal: random 16-of-32 rows rarely share a block
    gen = torch.Generator().manual_seed(11)
    q = torch.randn(1, 16, S, _D, generator=gen)
    k = torch.randn(1, 1, T, _D, generator=gen)
    v = torch.randn(1, 1, T, _D, generator=gen)
    idx = _local_indices(1, S, T, topk, 0, False, gen, dup_rate=0.3, tail_rate=0.5, scatter=True)
    packed = _run(device, monkeypatch, group, q, k, v, idx, chunk_start, ttnn.bfloat16)
    legacy = _run(device, monkeypatch, 0, q, k, v, idx, chunk_start, ttnn.bfloat16)
    assert not torch.isnan(packed).any()
    for s in range(S):
        ps = pcc(packed[:, :, s], legacy[:, :, s])
        assert ps > 0.999, f"token {s}: packed vs legacy pcc={ps}"
