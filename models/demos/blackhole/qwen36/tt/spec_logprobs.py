# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host helpers for device-side logprobs of the fused MTP step (work/research_1026/67, step S1.1). Pure torch, no ttnn.

The device gives, per verify row, the union of the per-shard top-64 raw logits (spec_fused_sampled.union_topk_stable)
and the sums L_raw = sum exp(x - M), L_T = sum exp((x - M) / T) over the whole vocabulary, where M is the row's global
maximum (the stats trace, MD._forward_verify_stats_body). From them the host builds a payload (``DeviceCandidates``):
the global top-64 (ids, raw logit values) and the raw logsumexp. The plugin turns the payload into vLLM
``LogprobsTensors`` (raw logprobs = log_softmax of the raw logits, vLLM 0.30 default ``logprobs_mode``).

Order everywhere: value descending, global id ascending among equal values (the order of the device topk and of the
host reference).
"""

from typing import NamedTuple

import torch

from models.demos.blackhole.qwen36.tt.spec_fused_sampled import union_topk_stable

NUM_CANDIDATES = 64  # M: per-shard top-M of the topk trace = size of the global candidate list


class DeviceCandidates(NamedTuple):
    """Per-user, per-verify-position payload. B users, T = K + 1 positions, M = 64 candidates."""

    ids: torch.Tensor  # int32 [B, T, M] global token ids, (value desc, id asc)
    values: torch.Tensor  # float32 [B, T, M] raw logits (bf16 values upcast)
    lse: torch.Tensor  # float32 [B, T] raw logsumexp over the whole vocabulary


def candidates_from_union(union_vals, union_local_ids, per_shard, M=NUM_CANDIDATES):
    """Global top-M of the topk trace's union: (values fp32 [R, M], ids int64 [R, M]). Same function as the sampled
    accept's step 1 (union_topk_stable), so a logprobs row and the accept see the same candidates."""
    return union_topk_stable(union_vals, union_local_ids, per_shard, M)


def lse_from_stats(max_val, l_raw):
    """Raw logsumexp in float64 from the row maximum M [R] and L_raw [R] = sum exp(x - M): M + log(L_raw)."""
    return torch.as_tensor(max_val).to(torch.float64) + torch.as_tensor(l_raw).to(torch.float64).log()


def combine_shard_sums(shard_sums):
    """Sum per-shard partial sums [S, R] (each taken relative to the SAME global maximum) in float32, the device's
    arithmetic (an fp32 all-gather + sum over the shard axis). Returns float32 [R]."""
    return torch.as_tensor(shard_sums).to(torch.float32).sum(dim=0)


def stats_from_logits(logits, temperature):
    """Float64 reference of the stats trace on FULL logits [R, V]: (M [R], L_raw [R], L_T [R]); L_T uses the row's
    temperature [R] (<= 0 or greedy rows pass 1.0). Used by tests and by the host fallback payload."""
    x = torch.as_tensor(logits).to(torch.float64)
    t = torch.as_tensor(temperature).reshape(-1, 1).to(torch.float64)
    m = x.max(dim=-1, keepdim=True).values
    d = x - m
    return m.squeeze(-1), d.exp().sum(-1), (d / t).exp().sum(-1)


def _stable_topm_row(row, M):
    """(values, ids) of the first M entries of ``row`` [V] under (value desc, id asc): topk for the threshold, then the
    whole tie group at the threshold, stable-sorted (torch.topk alone does not fix the order of ties)."""
    row = row.to(torch.float32)
    M = min(M, row.numel())
    thr = torch.topk(row, M).values[-1]
    ids = (row >= thr).nonzero().reshape(-1)  # ascending ids
    vals = row[ids]
    order = torch.sort(vals, descending=True, stable=True).indices[:M]
    return vals[order], ids[order]


def candidates_from_logits(logits, B, T, M=NUM_CANDIDATES):
    """The payload from FULL logits [>= B*T, V] (the engine fallback: mixed_rope_delta, eager_drafter, ...). Same
    contents as the device payload: top-M raw values and ids per row, float64 logsumexp rounded to float32."""
    x = torch.as_tensor(logits)[: B * T]
    vals = torch.empty(B * T, M, dtype=torch.float32)
    ids = torch.empty(B * T, M, dtype=torch.int64)
    for r in range(B * T):
        vals[r], ids[r] = _stable_topm_row(x[r], M)
    lse = torch.logsumexp(x.to(torch.float64), dim=-1).to(torch.float32)
    return make_candidates(vals, ids, lse, B, T)


def make_candidates(values, ids, lse, B, T):
    """Shape and dtype the payload: values [R, M] / ids [R, M] / lse [R] (R = B*T, row u*T + j) -> DeviceCandidates."""
    values, ids, lse = torch.as_tensor(values), torch.as_tensor(ids), torch.as_tensor(lse)
    R = B * T
    if values.shape[0] < R or values.shape != ids.shape or lse.numel() < R:
        raise ValueError(
            f"candidate shapes values {tuple(values.shape)} ids {tuple(ids.shape)} lse {tuple(lse.shape)}, R={R}"
        )
    M = values.shape[1]
    return DeviceCandidates(
        ids[:R].reshape(B, T, M).to(torch.int32),
        values[:R].reshape(B, T, M).to(torch.float32),
        lse.reshape(-1)[:R].reshape(B, T).to(torch.float32),
    )


def candidates_from_device(union_vals, union_local_ids, max_val, l_raw, per_shard, B, T, M=NUM_CANDIDATES):
    """The payload from the stats trace readback: union (vals [R', TP*M], local ids [R', TP*M]) and per-row L_raw = sum
    exp(x - M_row). ``max_val`` [R] is the row maximum the trace used (the max of the union row); None = take it from the
    union here (same value). L_raw >= 1 for a real row (the maximum term is exp(0)); a stale or zero buffer row (a parked row
    that no replay wrote) is clamped to 1, so the lse stays finite."""
    R = B * T
    uv = torch.as_tensor(union_vals)[:R]
    vals, gids = candidates_from_union(uv, torch.as_tensor(union_local_ids)[:R], per_shard, M)
    mx = vals[:, 0] if max_val is None else torch.as_tensor(max_val)[:R]
    lr = torch.as_tensor(l_raw)[:R].to(torch.float64).clamp(min=1.0)
    lse = lse_from_stats(mx, lr).to(torch.float32)
    return make_candidates(vals, gids, lse, B, T)


def permute_candidates(cand, perm):
    """Reorder users: row i of the result = user ``perm[i]`` of ``cand`` (the engine's _perm_slots direction)."""
    p = torch.as_tensor(perm).to(torch.int64)
    return DeviceCandidates(cand.ids[p], cand.values[p], cand.lse[p])


def gather_logprobs_from_candidates(cand_ids, cand_values, lse, chosen, num_logprobs):
    """vLLM ``Sampler.gather_logprobs`` layout from the payload, for N rows: column 0 = the chosen token, columns
    1..num_logprobs = the global top-N. cand_ids [N, M], cand_values [N, M] raw logits, lse [N], chosen [N] token ids.
    Returns (token_ids int64 [N, n+1], logprobs fp32 [N, n+1], rank int64 [N], lower_bound bool [N], missing bool [N]).
    rank = number of vocabulary tokens whose logprob is >= the chosen one (vLLM). It is exact when the M-th candidate
    value is strictly below the chosen value, else a lower bound (``lower_bound``). A chosen id outside the M candidates
    (impossible by the union proof) is ``missing``: logprob = the M-th candidate value (an upper bound), rank M + 1."""
    ids = torch.as_tensor(cand_ids).to(torch.int64)
    vals = torch.as_tensor(cand_values).to(torch.float32)
    lse = torch.as_tensor(lse).to(torch.float32).reshape(-1, 1)
    chosen = torch.as_tensor(chosen).to(torch.int64).reshape(-1, 1)
    N, M = ids.shape
    if not 0 <= num_logprobs <= M:
        raise ValueError(f"num_logprobs {num_logprobs} outside [0, {M}]")
    hit = ids == chosen
    missing = ~hit.any(dim=1)
    pos = hit.to(torch.int8).argmax(dim=1)
    cval = torch.where(missing, vals[:, M - 1], vals.gather(1, pos[:, None]).reshape(-1))
    rank = (vals >= cval[:, None]).sum(dim=1)
    rank = torch.where(missing, torch.full_like(rank, M + 1), rank)
    lower_bound = ~missing & (vals[:, M - 1] >= cval)
    tok = torch.cat([chosen, ids[:, :num_logprobs]], dim=1)
    lp = torch.cat([cval[:, None], vals[:, :num_logprobs]], dim=1) - lse
    return tok, lp, rank.to(torch.int64), lower_bound, missing
