# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Index algebra of the local token-sorted routed-expert prefill (``local_routed_prefill.py``), in plain torch.

The device path builds the same quantities with ttnn ops (``sort`` / ``masked_bincount`` / a prefix-sum matmul /
``scatter`` / ``embedding``); this module is its host reference so the index conventions can be unit-tested
without a device and the device test can compare against a routed (not masked-dense) golden.

Conventions (one chip, ``S`` tokens, ``K`` slots per token, ``E`` global experts, ``N = S * K`` pairs):

* pair ``p = t * K + k`` is token ``t``'s slot ``k`` with global expert ``gid[t, k]``;
* ``pos[b]`` is the pair at sorted position ``b`` (pairs sorted by global expert id, ties by pair index);
* ``counts[g]`` pairs route to expert ``g``; ``offsets[g]`` = number of pairs with an expert id below ``g``, i.e. the
  first row of expert ``g``'s region in the expert-major buffer;
* ``inv[p]`` is the sorted position of pair ``p``;
* the device buffer rounds every expert region up to a whole 32-row tile (the kernel addresses regions in
  tile-rows): ``aligned_offsets[g]`` is the exclusive prefix sum of the rounded counts and sorted row ``b`` of
  expert ``g`` lands at buffer row ``b + padshift[g]`` with ``padshift = aligned_offsets - offsets``.
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


def aligned_layout(indices: torch.Tensor, num_experts: int):
    """``indices`` [S, K] -> (aligned_offsets [E], dest [N] buffer row of every sorted row, buffer rows M)."""
    pos, counts, offsets, _ = index_pipeline(indices, num_experts)
    aligned_counts = (counts + TILE - 1) // TILE * TILE
    aligned_offsets = torch.cumsum(aligned_counts, dim=0) - aligned_counts
    padshift = aligned_offsets - offsets
    gid_sorted = indices.reshape(-1).to(torch.int64)[pos]
    dest = torch.arange(pos.numel(), dtype=torch.int64) + padshift[gid_sorted]
    return aligned_offsets, dest, pos.numel() + TILE * num_experts


def local_expert_mask(num_experts: int, base: int, num_local: int) -> torch.Tensor:
    """masked_bincount mask [E + 1] int32: 0 (counted) on this chip's experts ``[base, base + num_local)``, -1
    elsewhere and on the trailing sentinel column."""
    mask = torch.full((num_experts + 1,), -1, dtype=torch.int32)
    mask[base : base + num_local] = 0
    return mask


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
