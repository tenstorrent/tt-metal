# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 reference operations on explicit tensors: the damp gate and the two-bank sparse MLA.

Each op takes every input as a tensor so a device test can feed it the device's own inputs (teacher forcing).
All math runs in float32 and returns float32.

Two-bank sparse MLA (loop pass >= 2, assumptions A2-A6 in ``glm_5_4_config``) extends the GLM-5.3 op
``cpu_deepseek_v32.sparse_sdpa_prefill.sparse_mla`` to two key banks under one softmax:

    q         [1, H, S, 576]  absorbed pass-2 queries (q_nope . wkv_b ++ rotated q_pe)
    kvpe_a    [Ta, 576]       bank A: the pass-1 KVPE cache of the same layer
    indices_a [1, 1, S, ka]   pass-1 top-k positions into kvpe_a; 0xFFFFFFFF = masked
    kvpe_b    [Tb, 576]       bank B: pass-2 KVPE rows (window carry ++ this chunk)
    indices_b [1, 1, S, kb]   window rows into kvpe_b (see ``window_indices``); 0xFFFFFFFF = masked
    ->        [1, H, S, 512]  latent-space output (value absorb happens after the op)

Scores of both banks share one softmax with no sink or bias. A position selected in bank A and also inside the
window of bank B contributes two keys (pass-1 latent and pass-2 latent).
"""

import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.sparse_sdpa_prefill import MASKED_INDEX

_ROW_BLOCK = 256  # query rows per block of two_bank_sparse_mla


def damp_gate_reference(
    x_prev: torch.Tensor,
    h: torch.Tensor,
    *,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Damp gate after a loop pass (A1): x_k = x_{k-1} + beta * (h_k - x_{k-1}).

    beta = sigmoid(RMSNorm(x_{k-1}; norm_weight, eps) @ proj_weight^T + proj_bias); the gate reads the pass
    input ``x_prev``, not the pass output ``h``.

    Args:
        x_prev: pass input x_{k-1} [..., hidden].
        h: pass output h_k [..., hidden].
        norm_weight: RMSNorm gain [hidden].
        proj_weight: gate Linear weight [width, hidden]; width 1 = one beta per token, width hidden = per channel.
        proj_bias: gate Linear bias [width].
        eps: RMSNorm epsilon.

    Returns:
        (x_k, beta), float32; beta is [..., width].
    """
    hidden = x_prev.shape[-1]
    width = proj_weight.shape[0]
    if h.shape != x_prev.shape:
        raise ValueError(f"pass output {tuple(h.shape)} != pass input {tuple(x_prev.shape)}")
    if proj_weight.shape != (width, hidden) or width not in (1, hidden):
        raise ValueError(f"gate weight must be [1, {hidden}] or [{hidden}, {hidden}], got {tuple(proj_weight.shape)}")
    if proj_bias.shape != (width,) or norm_weight.shape != (hidden,):
        raise ValueError(f"gate bias {tuple(proj_bias.shape)} / norm {tuple(norm_weight.shape)} do not match width")
    x = x_prev.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_weight.float()
    beta = torch.sigmoid(normed @ proj_weight.float().t() + proj_bias.float())
    return x + beta * (h.float() - x), beta


def window_indices(q_start: int, seq_len: int, window: int, *, kv_start: int = 0) -> torch.Tensor:
    """Bank-B indices for query positions q_start..q_start+seq_len-1: row p holds p-window+1 .. p (A3).

    Indices address a bank-B buffer whose row 0 is absolute position ``kv_start`` (e.g. a chunk's window
    carry). Positions before 0 are sentinels. Returns [1, 1, seq_len, window] int64, ascending positions.
    """
    p = torch.arange(q_start, q_start + seq_len).unsqueeze(-1)
    pos = p - window + 1 + torch.arange(window).unsqueeze(0)  # [S, W]
    needed = pos[pos >= 0]
    if needed.numel() and int(needed.min()) < kv_start:
        raise ValueError(f"window needs position {int(needed.min())} but bank B starts at {kv_start}")
    idx = torch.where(pos >= 0, pos - kv_start, torch.full_like(pos, MASKED_INDEX))
    return idx.view(1, 1, seq_len, window)


def pass1_indices(topk: torch.Tensor, q_start: int, index_topk: int) -> torch.Tensor:
    """Pass-1 top-k in the device form: [1, 1, S, index_topk], non-causal and missing slots 0xFFFFFFFF (A5).

    The CPU indexer returns ``min(index_topk, end)`` positions per row and relies on the causal mask for rows
    with fewer causal keys than that; the device fills those slots, and the slots beyond ``end``, with the
    sentinel instead. ``topk`` is [1, S, k] or [1, 1, S, k], k <= index_topk, for query rows starting at
    absolute position ``q_start``.
    """
    seq_len, k = topk.shape[-2], topk.shape[-1]
    if k > index_topk:
        raise ValueError(f"top-k width {k} exceeds index_topk {index_topk}")
    idx = torch.full((1, 1, seq_len, index_topk), MASKED_INDEX, dtype=torch.int64)
    idx[..., :k] = topk.reshape(1, 1, seq_len, k).to(torch.int64)
    pos = torch.arange(q_start, q_start + seq_len).view(1, 1, -1, 1)
    return torch.where(idx > pos, torch.full_like(idx, MASKED_INDEX), idx)


def _gather_bank(kvpe: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one bank's selected rows: kvpe [T, D], indices [1, 1, S, k] -> rows [S, k, D], masked [S, k]."""
    idx = indices.reshape(indices.shape[-2], indices.shape[-1]).to(torch.int64)
    masked = idx == MASKED_INDEX
    valid = idx[~masked]
    if valid.numel() and (int(valid.min()) < 0 or int(valid.max()) >= kvpe.shape[0]):
        raise IndexError(f"bank index out of range [0, {kvpe.shape[0]}): [{int(valid.min())}, {int(valid.max())}]")
    rows = kvpe.float()[torch.where(masked, torch.zeros_like(idx), idx)]
    return rows, masked


def two_bank_sparse_mla(
    q: torch.Tensor,
    kvpe_a: torch.Tensor,
    indices_a: torch.Tensor,
    kvpe_b: torch.Tensor,
    indices_b: torch.Tensor,
    scale: float,
    *,
    latent_dim: int,
) -> torch.Tensor:
    """One softmax over bank A (pass-1 KVPE at pass-1 top-k) and bank B (pass-2 KVPE window) (A6).

    Shapes in the module docstring; ``latent_dim`` is kv_lora_rank (512 for GLM), the V width of each key row.
    A row needs at least one unmasked key: there is no sink, so a row with none is undefined (NaN). Every real
    row has one, position p itself in bank B.
    """
    if q.dim() != 4 or q.shape[0] != 1:
        raise ValueError(f"q must be [1, H, S, D], got {tuple(q.shape)}")
    _, n_heads, seq_len, width = q.shape
    for name, kvpe, idx in (("a", kvpe_a, indices_a), ("b", kvpe_b, indices_b)):
        if kvpe.dim() != 2 or kvpe.shape[-1] != width:
            raise ValueError(f"kvpe_{name} must be [T, {width}], got {tuple(kvpe.shape)}")
        if idx.shape[-2] != seq_len:
            raise ValueError(f"indices_{name} rows {idx.shape[-2]} != query rows {seq_len}")
    out = []
    # Query rows are independent; blocks bound the gathered keys ([rows, ka + kb, D] float32) to ~1 GB at production
    # sizes (5120 rows x 2176 keys x 576 would be 26 GB at once). Same math per row.
    block = _ROW_BLOCK
    for lo in range(0, seq_len, block):
        hi = min(lo + block, seq_len)
        rows_a, masked_a = _gather_bank(kvpe_a, indices_a[..., lo:hi, :])
        rows_b, masked_b = _gather_bank(kvpe_b, indices_b[..., lo:hi, :])
        rows = torch.cat([rows_a, rows_b], dim=1)  # [s, ka + kb, D]
        masked = torch.cat([masked_a, masked_b], dim=1)  # [s, ka + kb]

        scores = torch.einsum("hsd,sjd->hsj", q[0, :, lo:hi].float(), rows) * scale  # [H, s, ka + kb]
        scores = scores.masked_fill(masked.unsqueeze(0), float("-inf"))
        probs = scores.softmax(dim=-1)
        out.append(torch.einsum("hsj,sjc->hsc", probs, rows[..., :latent_dim]))
    return torch.cat(out, dim=1).unsqueeze(0)  # [1, H, S, latent_dim]
