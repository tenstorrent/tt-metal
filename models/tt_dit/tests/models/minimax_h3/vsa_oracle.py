# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Torch oracle for the VSA attention path (R3/R4/R6 of VSA_SCOPE.md).

Operates in the geometry's placement order on zero-padded tiled tensors
(``[B, H, S_pad, D]``, ``S_pad = n_tiles * 64``): exactly the tensors the
device path sees. Semantics mirror FastVideo's VSA-H3 backend (vendored under
``models/tt_dit/models/transformers/minimax_h3/vsa_reference/``), extended
with pad tiles: zero-valid tiles are never listed in any index row and their
query rows produce don't-care outputs (dropped at unpacking).

Index-tensor contract (matches R4's ``vsa_sdpa``):
- shape ``[1, H, n_q_tiles, n_tiles]`` uint32, one row per (head, 64-token
  query tile), values are placement-order tile ids, ``0xFFFFFFFF`` sentinel
  tail;
- an exempt-query row lists every real (valid_count > 0) tile id;
- every other row lists ``[all exempt tile ids] + [its top-k candidate ids]``,
  k = max(1, min(ceil((1 - sparsity) * n_candidates), n_candidates)).
"""

from __future__ import annotations

import math

import torch

from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, MiniMaxH3VSAGeometry

VSA_INDEX_SENTINEL = 0xFFFFFFFF


def pool_tiles(x_tiled: torch.Tensor, geometry: MiniMaxH3VSAGeometry) -> torch.Tensor:
    """Masked mean per tile. ``x_tiled``: [B, H, S_pad, D] -> [B, H, n_tiles, D] fp32.

    Uses the host-built averaging matrix (zero columns at pad slots, zero rows
    for pad tiles), so pad-slot values never contribute -- the device coarse
    stage pools with this same matrix.
    """
    matrix = geometry.averaging_matrix().to(x_tiled.device)
    return torch.einsum("ts,bhsd->bhtd", matrix, x_tiled.to(torch.float32))


def coarse_scores(q_tiled: torch.Tensor, k_tiled: torch.Tensor, geometry: MiniMaxH3VSAGeometry) -> torch.Tensor:
    """[B, H, n_tiles, n_tiles] fp32: pooled-Q @ pooled-K^T / sqrt(head_dim)."""
    q_pooled = pool_tiles(q_tiled, geometry)
    k_pooled = pool_tiles(k_tiled, geometry)
    return torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / (q_tiled.shape[-1] ** 0.5)


def compute_topk(sparsity: float, num_candidates: int) -> int:
    """Candidate tiles to keep; FastVideo's ``compute_topk``, clamped to [1, n]."""
    return max(1, min(math.ceil((1 - sparsity) * num_candidates), num_candidates))


def select_index_rows(
    scores: torch.Tensor, geometry: MiniMaxH3VSAGeometry, sparsity: float
) -> tuple[torch.Tensor, list[list[set[int]]]]:
    """Index tensor per the R4 contract, plus per-row python sets for set-wise checks.

    ``scores``: [B, H, n_tiles, n_tiles] (query tile x key tile). Returns the
    uint32 index tensor [B, H, n_tiles, n_tiles] (sentinel-tailed, unsorted
    top-k order after the exempt prefix) and ``sets[h][row]`` for B = 1.
    """
    batch, heads, n_tiles, _ = scores.shape
    assert n_tiles == geometry.n_tiles
    exempt_ids = torch.nonzero(geometry.is_exempt, as_tuple=False).reshape(-1)
    candidate_ids = torch.nonzero(geometry.is_candidate, as_tuple=False).reshape(-1)
    real_ids = torch.nonzero(geometry.valid_counts > 0, as_tuple=False).reshape(-1)
    k = compute_topk(sparsity, int(candidate_ids.numel()))

    indices = torch.full((batch, heads, n_tiles, n_tiles), VSA_INDEX_SENTINEL, dtype=torch.int64)
    sets: list[list[set[int]]] = []
    for b in range(batch):
        head_sets: list[list[set[int]]] = []
        for h in range(heads):
            row_sets: list[set[int]] = []
            for row in range(n_tiles):
                if bool(geometry.is_exempt[row]):
                    listed = real_ids
                else:
                    top = scores[b, h, row, candidate_ids].topk(k).indices
                    listed = torch.cat([exempt_ids, candidate_ids[top]])
                indices[b, h, row, : listed.numel()] = listed
                row_sets.append(set(listed.tolist()))
            head_sets.append(row_sets)
        if b == 0:
            sets = head_sets
    return indices.to(torch.uint32), sets


def coarse_output(
    scores: torch.Tensor, v_tiled: torch.Tensor, geometry: MiniMaxH3VSAGeometry, dtype: torch.dtype
) -> torch.Tensor:
    """O_c broadcast tile -> 64 tokens: [B, H, S_pad, D] = softmax(scores) @ pooled-V."""
    v_pooled = pool_tiles(v_tiled, geometry)
    out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)  # [B, H, n_tiles, D]
    return out_c.to(dtype).repeat_interleave(VSA_TILE_TOKENS, dim=2)


def fine_attention(
    q_tiled: torch.Tensor,
    k_tiled: torch.Tensor,
    v_tiled: torch.Tensor,
    indices: torch.Tensor,
    block_counts: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference for R4's ``vsa_sdpa``: exact attention over the listed blocks.

    ``q/k/v``: [B, H, S, D] with S (and the K/V length T) multiples of 64;
    ``indices``: [B, H, S/64, T/64] uint32 with sentinel tails; ``block_counts``:
    [T/64] valid tokens per block. Geometry-free on purpose -- the kernel's
    whole contract is the index list, so this reference is too.
    """
    batch, heads, seq_len, dim = q_tiled.shape
    n_kv_blocks = k_tiled.shape[2] // VSA_TILE_TOKENS
    scale = dim**-0.5 if scale is None else scale
    counts = block_counts.to(torch.long)

    token_valid = (torch.arange(VSA_TILE_TOKENS)[None, :] < counts[:, None]).reshape(-1)  # [T]
    out = torch.zeros_like(q_tiled)
    for b in range(batch):
        for h in range(heads):
            for q_tile in range(seq_len // VSA_TILE_TOKENS):
                row = indices[b, h, q_tile].to(torch.long)
                listed = row[row != VSA_INDEX_SENTINEL]
                if listed.numel() == 0:
                    continue
                assert bool((listed < n_kv_blocks).all())
                cols = (listed[:, None] * VSA_TILE_TOKENS + torch.arange(VSA_TILE_TOKENS)[None, :]).reshape(-1)
                keep = token_valid[cols]
                cols = cols[keep]
                rows = slice(q_tile * VSA_TILE_TOKENS, (q_tile + 1) * VSA_TILE_TOKENS)
                attn = torch.einsum("qd,kd->qk", q_tiled[b, h, rows].float(), k_tiled[b, h, cols].float()) * scale
                out[b, h, rows] = (torch.softmax(attn, dim=-1) @ v_tiled[b, h, cols].float()).to(q_tiled.dtype)
    return out


def vsa_attention(
    q_tiled: torch.Tensor,
    k_tiled: torch.Tensor,
    v_tiled: torch.Tensor,
    geometry: MiniMaxH3VSAGeometry,
    sparsity: float,
    gate_tiled: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full VSA oracle in tiled order: fine attention + gated coarse branch.

    All tensors [B, H, S_pad, D] in placement order, zeros at pad slots.
    Output rows of pad slots are don't-cares.
    """
    scores = coarse_scores(q_tiled, k_tiled, geometry)
    indices, _ = select_index_rows(scores, geometry, sparsity)
    block_counts = geometry.valid_counts.to(torch.uint32)
    out = fine_attention(q_tiled, k_tiled, v_tiled, indices, block_counts)
    if gate_tiled is not None:
        out = out + gate_tiled * coarse_output(scores, v_tiled, geometry, out.dtype)
    return out
