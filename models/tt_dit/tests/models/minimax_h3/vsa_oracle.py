# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Torch oracle for the VSA attention path.

Operates on ``[B, H, padded_len, D]`` tensors in the geometry's device row
order, zero at pad rows: exactly the tensors the device path sees. Semantics
mirror FastVideo's VSA-H3 backend (vendored under ``vsa_reference/`` next to
this file), extended with pad blocks: count-0 blocks are never listed in any
index row and their query rows produce don't-care outputs (dropped at
unpacking).

Index-tensor contract:
- shape ``[1, H, n_blocks, n_blocks]`` uint32, one row per (head, query
  block), values are global block ids, ``0xFFFFFFFF`` sentinel tail;
- an exempt-query row lists every real (valid_count > 0) block;
- every other row lists ``[all exempt blocks] + [its top-k candidate blocks]``,
  k = max(1, min(ceil((1 - sparsity) * n_candidates), n_candidates)).
"""

from __future__ import annotations

import torch

from ....models.transformers.minimax_h3.vsa_prepass import compute_topk
from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, MiniMaxH3VSAGeometry

VSA_INDEX_SENTINEL = 0xFFFFFFFF


def pool_tiles(x: torch.Tensor, geometry: MiniMaxH3VSAGeometry) -> torch.Tensor:
    """Masked mean per block. ``x``: [B, H, padded_len, D] -> [B, H, n_blocks, D] fp32.

    Uses the host-built averaging matrix (zero weight on pad rows and pad
    blocks), so pad-row values never contribute -- the device coarse stage
    pools with this same matrix.
    """
    matrix = geometry.averaging_matrix().to(x.device)
    return torch.einsum("ts,bhsd->bhtd", matrix, x.to(torch.float32))


def coarse_scores(q: torch.Tensor, k: torch.Tensor, geometry: MiniMaxH3VSAGeometry) -> torch.Tensor:
    """[B, H, n_blocks, n_blocks] fp32: pooled-Q @ pooled-K^T / sqrt(head_dim)."""
    q_pooled = pool_tiles(q, geometry)
    k_pooled = pool_tiles(k, geometry)
    return torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / (q.shape[-1] ** 0.5)


def select_index_rows(
    scores: torch.Tensor, geometry: MiniMaxH3VSAGeometry, sparsity: float, with_sets: bool = True
) -> tuple[torch.Tensor, list[list[set[int]]] | None]:
    """Index tensor per the contract above, plus per-row python sets for set-wise checks.

    ``scores``: [B, H, n_blocks, n_blocks] (query block x key block). Returns
    the uint32 index tensor of the same shape (sentinel-tailed, unsorted top-k
    order after the exempt prefix) and, when ``with_sets``, ``sets[h][row]``
    for B = 1 (skip at production scale -- the sets loop is per-row).
    """
    batch, heads, n_blocks, _ = scores.shape
    assert n_blocks == geometry.n_blocks
    exempt_ids = torch.nonzero(geometry.is_exempt, as_tuple=False).reshape(-1)
    candidate_ids = torch.nonzero(geometry.is_candidate, as_tuple=False).reshape(-1)
    real_ids = torch.nonzero(geometry.valid_counts > 0, as_tuple=False).reshape(-1)
    k = compute_topk(sparsity, int(candidate_ids.numel()))

    # Candidate rows, vectorized: [exempt ids | top-k candidate ids | sentinel tail].
    top_in_candidates = scores[..., candidate_ids].topk(k, dim=-1).indices
    listed = torch.cat(
        [
            exempt_ids.reshape(1, 1, 1, -1).expand(batch, heads, n_blocks, -1),
            candidate_ids[top_in_candidates],
            torch.full(
                (batch, heads, n_blocks, n_blocks - exempt_ids.numel() - k), VSA_INDEX_SENTINEL, dtype=torch.int64
            ),
        ],
        dim=-1,
    )
    # Exempt-query rows take the dense list of all real blocks.
    dense_row = torch.full((n_blocks,), VSA_INDEX_SENTINEL, dtype=torch.int64)
    dense_row[: real_ids.numel()] = real_ids
    indices = torch.where(geometry.is_exempt.reshape(1, 1, -1, 1), dense_row.reshape(1, 1, 1, -1), listed)

    sets = None
    if with_sets:
        sets = [
            [set(row[row != VSA_INDEX_SENTINEL].tolist()) for row in (indices[0, h, r] for r in range(n_blocks))]
            for h in range(heads)
        ]
    return indices.to(torch.uint32), sets


def coarse_output(
    scores: torch.Tensor, v: torch.Tensor, geometry: MiniMaxH3VSAGeometry, dtype: torch.dtype
) -> torch.Tensor:
    """O_c broadcast block -> its rows: [B, H, padded_len, D] = softmax(scores) @ pooled-V.

    Pad-block columns are masked out of the softmax, as the device pre-pass does: their zero pooled keys would
    otherwise score 0 and dilute O_c. FastVideo has no pad blocks, so this changes nothing there.
    """
    v_pooled = pool_tiles(v, geometry)
    scores = scores.masked_fill(geometry.valid_counts == 0, -float("inf"))
    out_per_block = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)
    return out_per_block.to(dtype)[:, :, geometry.block_of_row]


def fine_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    geometry: MiniMaxH3VSAGeometry,
    scale: float | None = None,
) -> torch.Tensor:
    """Block-sparse reference: exact attention over the listed blocks.

    ``q/k/v``: [B, H, padded_len, D] in device row order; ``indices``:
    [B, H, n_blocks, n_blocks] uint32 with sentinel tails. A listed block
    contributes its first ``valid_counts`` rows.
    """
    batch, heads, _, dim = q.shape
    scale = dim**-0.5 if scale is None else scale
    starts = geometry.block_starts.tolist()
    counts = geometry.valid_counts.tolist()
    block_rows = torch.bincount(geometry.block_of_row, minlength=geometry.n_blocks).tolist()

    out = torch.zeros_like(q)
    for b in range(batch):
        for h in range(heads):
            for q_block in range(geometry.n_blocks):
                row = indices[b, h, q_block].to(torch.long)
                listed = row[row != VSA_INDEX_SENTINEL].tolist()
                if not listed:
                    continue
                cols = torch.cat([torch.arange(starts[j], starts[j] + counts[j]) for j in listed])
                rows = slice(starts[q_block], starts[q_block] + block_rows[q_block])
                attn = torch.einsum("qd,kd->qk", q[b, h, rows].float(), k[b, h, cols].float()) * scale
                out[b, h, rows] = (torch.softmax(attn, dim=-1) @ v[b, h, cols].float()).to(q.dtype)
    return out


def fine_attention_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    geometry: MiniMaxH3VSAGeometry,
    scale: float | None = None,
    row_chunk: int = 64,
) -> torch.Tensor:
    """`fine_attention`, vectorized over chunks of query blocks for production-scale shapes.

    Same contract and results (up to fp32 reduction order); gathers each chunk's listed blocks into a padded
    [chunk, width * 64] key window and masks pad columns, so memory stays bounded by ``row_chunk * width * 64``.
    """
    batch, heads, seq_len, dim = q.shape
    scale = dim**-0.5 if scale is None else scale
    n_blocks = geometry.n_blocks
    starts = geometry.block_starts
    counts = geometry.valid_counts
    block_rows = torch.bincount(geometry.block_of_row, minlength=n_blocks)
    arange64 = torch.arange(VSA_TILE_TOKENS)

    idx = indices.to(torch.int64)
    is_listed = idx != VSA_INDEX_SENTINEL
    out = torch.zeros_like(q)
    for b in range(batch):
        for h in range(heads):
            for r0 in range(0, n_blocks, row_chunk):
                chunk = slice(r0, min(r0 + row_chunk, n_blocks))
                width = int(is_listed[b, h, chunk].sum(dim=-1).max())
                if width == 0:
                    continue
                key_blocks = idx[b, h, chunk, :width].clamp(min=0, max=n_blocks - 1)
                listed = is_listed[b, h, chunk, :width]
                # Rows past a block's count are masked, so clamping a short block's tail row index is harmless.
                key_rows = (starts[key_blocks][:, :, None] + arange64).clamp(max=seq_len - 1)
                key_ok = listed[:, :, None] & (arange64 < counts[key_blocks][:, :, None])
                key_rows, key_ok = key_rows.flatten(1), key_ok.flatten(1)
                query_rows = (starts[chunk][:, None] + arange64).clamp(max=seq_len - 1)
                query_ok = arange64 < block_rows[chunk][:, None]

                keys = k[b, h][key_rows].float()
                vals = v[b, h][key_rows].float()
                queries = q[b, h][query_rows].float()
                attn = torch.einsum("rqd,rkd->rqk", queries, keys) * scale
                attn = attn.masked_fill(~key_ok[:, None, :], float("-inf"))
                res = torch.einsum("rqk,rkd->rqd", torch.softmax(attn, dim=-1), vals)
                has_keys = listed.any(dim=-1)[:, None] & query_ok
                out[b, h][query_rows[has_keys]] = res[has_keys].to(out.dtype)
    return out


def vsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    geometry: MiniMaxH3VSAGeometry,
    sparsity: float,
    gate: torch.Tensor | None = None,
    batched: bool = False,
) -> torch.Tensor:
    """Full VSA oracle in device row order: fine attention + gated coarse branch.

    All tensors [B, H, padded_len, D], zeros at pad rows. Output pad rows are
    don't-cares. ``batched=True`` uses the chunk-vectorized fine attention
    (same results; production scale).
    """
    scores = coarse_scores(q, k, geometry)
    indices, _ = select_index_rows(scores, geometry, sparsity, with_sets=False)
    fine = fine_attention_batched if batched else fine_attention
    out = fine(q, k, v, indices, geometry)
    if gate is not None:
        out = out + gate * coarse_output(scores, v, geometry, out.dtype)
    return out
