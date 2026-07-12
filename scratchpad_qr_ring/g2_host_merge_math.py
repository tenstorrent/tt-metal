# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G2 (math half) — host online-softmax merge of per-shard sparse-MLA partials.

Proves the RImplementation of the cross-SP flash-merge BEFORE any fabric exists:
split the top-k selected indices into `sp` block-cyclic shards, compute each
shard's UNNORMALIZED partial (O_unnorm, m, l) with the exact op math, then merge
with online softmax and check it reproduces the full-T golden.

This is the reduce formula the device partial-export + fabric collective are later
checked against. Representation decision (Phase-0): raw (m,l), fp32 stats.

Merge math (raw-max online softmax), for shards i with local raw-max m_i, local
denom l_i, local unnormalized O_i = sum_j exp((s_j - m_i)*scale) * V_j:
    M      = max_i m_i
    a_i    = exp((m_i - M) * scale)          # correction, scale applied here (raw m)
    O      = sum_i a_i * O_i
    L      = sum_i a_i * l_i
    out    = O / L
Empty shard (no local hits) contributes the identity partial (O=0, m=-inf, l=0).
"""
import sys

import torch

sys.path.insert(0, "tests/ttnn/unit_tests/operations/sdpa")
from sparse_sdpa_test_utils import MASKED_INDEX, golden, make_inputs  # noqa: E402

K_DIM, V_DIM = 576, 512


def blockcyclic_shard_of(pos, sp, chunk_local, T):
    """Which SP shard physically holds natural token `pos` under block-cyclic layout.
    Mirrors mla/utils.py::blockcyclic_positions inverse: shard = (pos // chunk_local) % sp."""
    return (pos // chunk_local) % sp


def shard_partial(q, kv, indices, scale, shard, sp, chunk_local, T):
    """Unnormalized (O,m,l) for one SP shard: only the selected indices that live on
    `shard` participate; everything else is masked. Returns O[1,H,S,V], m[1,H,S,1], l[1,H,S,1]."""
    B, H, S, Dk = q.shape
    k = indices.shape[-1]
    idx = indices.reshape(B, S, k)
    masked = idx == MASKED_INDEX
    idx_safe = torch.where(masked, torch.zeros_like(idx), idx).to(torch.int64)
    # Keep only slots whose token lives on this shard AND is not a sentinel.
    on_shard = (blockcyclic_shard_of(idx_safe, sp, chunk_local, T) == shard) & (~masked)

    kv2 = kv[0, 0]  # [T, Dk]
    sel = kv2[idx_safe.reshape(-1)].reshape(B, S, k, Dk)
    scores = torch.einsum("bhsd,bsjd->bhsj", q, sel) * scale  # [B,H,S,k]
    neg = torch.finfo(torch.float32).min
    scores = scores.masked_fill(~on_shard.view(B, 1, S, k), neg)  # off-shard -> -inf

    m = scores.max(dim=-1, keepdim=True).values  # [B,H,S,1] raw max (already *scale here)
    # If a row has NO on-shard hit, max == neg; force the identity partial (m=-inf,l=0,O=0).
    empty = ~on_shard.any(dim=-1)  # [B,S]
    p = torch.exp(scores - m)  # exp(scale*(s - max)); note scores already include *scale
    p = p.masked_fill(~on_shard.view(B, 1, S, k), 0.0)
    l = p.sum(dim=-1, keepdim=True)  # [B,H,S,1]
    O = torch.einsum("bhsj,bsjd->bhsd", p, sel[..., :V_DIM])  # [B,H,S,V]
    # identity for empty rows
    m = m.masked_fill(empty.view(B, 1, S, 1), float("-inf"))
    l = l.masked_fill(empty.view(B, 1, S, 1), 0.0)
    O = O.masked_fill(empty.view(B, 1, S, 1), 0.0)
    return O, m, l


def merge(partials, stats_dtype=torch.float32):
    """Online-softmax merge of a list of (O,m,l) per-shard partials -> normalized out [1,H,S,V]."""
    ms = torch.stack([p[1].to(stats_dtype) for p in partials])  # [sp,B,H,S,1]
    M = ms.max(dim=0).values  # global raw max
    O = None
    L = 0.0
    for Oi, mi, li in partials:
        a = torch.exp((mi.to(stats_dtype) - M))  # scale already folded into m units
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)  # m=-inf -> a=0 (empty shard)
        contrib_O = a * Oi.to(stats_dtype)
        O = contrib_O if O is None else O + contrib_O
        L = L + a * li.to(stats_dtype)
    return O / L


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def run(H, S, T, TOPK, sp, chunk_local, stats_dtype, seed=0):
    scale = K_DIM**-0.5
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 10**9, seed=seed)  # all-valid
    ref = golden(q, kv, indices, scale, V_DIM)  # full-T normalized golden [1,H,S,V]
    partials = [shard_partial(q, kv, indices, scale, sh, sp, chunk_local, T) for sh in range(sp)]
    out = merge(partials, stats_dtype=stats_dtype)
    return pcc(out, ref)


def run_with_empty_shard(H, S, T, TOPK, sp, chunk_local, seed=1):
    """Force some query rows to select tokens from only a subset of shards (zero-hit shards present)."""
    scale = K_DIM**-0.5
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 3, seed=seed)  # only 3 valid/row -> sparse shards
    ref = golden(q, kv, indices, scale, V_DIM)
    partials = [shard_partial(q, kv, indices, scale, sh, sp, chunk_local, T) for sh in range(sp)]
    out = merge(partials, stats_dtype=torch.float32)
    return pcc(out, ref)


def main():
    print("G2 host online-softmax merge of per-shard partials (== full-T golden?)\n")
    print(f"{'sp':>4} {'T':>8} {'TOPK':>6} {'stats':>7} {'case':>14} {'PCC':>10}   gate>=0.99")
    print("-" * 66)
    for sp in [2, 4, 8]:
        T = sp * 256
        chunk_local = 32
        for stats_dtype, tag in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
            p = run(H=32, S=64, T=T, TOPK=64, sp=sp, chunk_local=chunk_local, stats_dtype=stats_dtype)
            ok = "PASS" if p >= 0.99 else "FAIL"
            print(f"{sp:>4} {T:>8} {64:>6} {tag:>7} {'all_valid':>14} {p:>10.6f}   {ok}")
        pe = run_with_empty_shard(H=32, S=64, T=T, TOPK=64, sp=sp, chunk_local=chunk_local)
        ok = "PASS" if pe >= 0.99 else "FAIL"
        print(f"{sp:>4} {T:>8} {64:>6} {'fp32':>7} {'zero-hit-shards':>14} {pe:>10.6f}   {ok}")
        print()


if __name__ == "__main__":
    main()
