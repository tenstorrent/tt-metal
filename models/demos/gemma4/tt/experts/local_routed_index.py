# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host reference of one chip's routed pairs for the local routed prefill (``local_routed_prefill.py``), in plain torch:
``local_routed_golden`` is the per-chip partial (the sum over the slots of this chip's experts) the device tests compare
against, computed per expert over its token region (``index_pipeline``: pairs sorted by expert id).

Conventions (one chip, ``S`` tokens, ``K`` slots per token, ``E`` global experts, ``N = S * K`` pairs):

* pair ``p = t * K + k`` is token ``t``'s slot ``k`` with global expert ``gid[t, k]``;
* ``pos[b]`` is the pair at sorted position ``b`` (pairs sorted by global expert id, ties by pair index);
* ``counts[g]`` pairs route to expert ``g``; ``offsets[g]`` = number of pairs with an expert id below ``g``;
* ``inv[p]`` is the sorted position of pair ``p``.

Why the first local path failed (2026-11-17 probes, _logs/lroute_1117_2002 / lroute_1117_2056 on l2-35): it built
these indices on device with ``ttnn.sort`` / ``masked_bincount`` / ``scatter`` / ``embedding`` and took the region
offsets (exclusive prefix sums of every expert's pair count, raw and tile-aligned) from an fp32 ``ttnn.matmul`` with a
strictly-upper-triangular ones matrix (HiFi4, fp32 dest). That matmul is exact while every expert holds at most 2047
pairs of the chunk and goes wrong by 1-2 rows once one holds 2048 or more (2048, 2169, 2171, 2173 wrong; 3056 exact by
its bit pattern): every later expert's region then starts in the wrong row, ~85% of the pairs read another token's
row (sort, bincount, scatter and the embedding gathers were exact at every size probed), and end to end at 4096 rows
with a 3718-row expert the chips after the hot one fell to partial PCC 0.88 vs this golden. Uniform test routing
(~S * K / E rows per expert) never comes near 2048 rows; the real router puts up to 2008 of a 2K prompt's 2048 tokens
on one expert, so 4K chunks of real prompts cross it. The 2K TTFT gate's 0.522 itself was not this bug (the 2K
counts stay below 2048 and the pipeline is exact there): that last position is ill-conditioned for every TT path
(the HF fp32 reference: dispatch 0.510, the new local path 0.555, its dense-routing arm 0.516), and the first path's
token there (1852) is HF's top-1. The current path computes the offsets with integer sums in the gather kernel.
"""

import torch

TILE = 32


def index_pipeline(indices: torch.Tensor, num_experts: int):
    """``indices`` [S, K] integer expert ids -> (pos [N], counts [E], offsets [E], inv [N]) as int64 tensors."""
    S, K = indices.shape
    flat = indices.reshape(-1).to(torch.int64)
    pos = torch.sort(flat, stable=True).indices  # pair index at every sorted position
    counts = torch.bincount(flat, minlength=num_experts)[:num_experts]
    offsets = torch.cumsum(counts, dim=0) - counts  # exclusive prefix sum
    inv = torch.empty_like(pos)
    inv[pos] = torch.arange(pos.numel(), dtype=torch.int64)
    return pos, counts, offsets, inv


def local_routed_golden(x, values, indices, base, num_local, gate_projs, up_projs, down_projs):
    """One chip's partial sum of the routed experts: ``sum_k w[t,k] * expert_{gid[t,k]}(x[t])`` over the slots whose
    expert lives on this chip (``[base, base + num_local)``), computed per expert over its sorted token region
    exactly as the device buffer is laid out (so a mismatch in the index algebra shows up here, not only in the
    final PCC). ``x`` [S, H]; ``values`` / ``indices`` [S, K]; weights per local slot in the device (K, N) /
    (N, K) orientation (``gate`` [H, I], ``up`` [H, I], ``down`` [I, H]). Returns [S, H] float32."""
    S, K = indices.shape
    E = int(indices.max().item()) + 1 if indices.numel() else num_local
    E = max(E, base + num_local)
    pos, counts, offsets, inv = index_pipeline(indices, E)
    xf = x.float()
    out = torch.zeros(S, xf.shape[-1], dtype=torch.float32)
    tok_of_pos = pos // K
    for e in range(num_local):
        g = base + e
        n, s = int(counts[g]), int(offsets[g])
        if n == 0:
            continue
        rows = tok_of_pos[s : s + n]
        xe = xf[rows]
        h = torch.nn.functional.gelu(xe @ gate_projs[e].float(), approximate="tanh") * (xe @ up_projs[e].float())
        ye = h @ down_projs[e].float()  # [n, H] in sorted order
        pairs = pos[s : s + n]  # pair index of every row
        w = values.reshape(-1).float()[pairs]
        out.index_add_(0, rows, ye * w[:, None])
    return out
