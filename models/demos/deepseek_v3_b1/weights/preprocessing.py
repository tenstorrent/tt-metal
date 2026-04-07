# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic **preprocess** steps: pure torch transforms from HF tensors to upload-ready tensors."""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.weights.types import (
    _KV_B_PROJ_HEAD_DIM,
    _KV_LORA_RANK,
    _MTP_NUM_DRAM_BANKS,
    _Q_HEAD_DIM,
    _QK_NOPE_HEAD_DIM,
)


def deinterleave_q_b_proj(q_b_proj: torch.Tensor, num_heads: int | None = None) -> torch.Tensor:
    """Convert q_b_proj.weight from HF interleaved to [ALL_NOPE | ALL_ROPE] layout.

    HF stores q_b_proj with out_features = num_heads * q_head_dim, where each head's
    nope and rope dims are contiguous: [h0_nope|h0_rope|h1_nope|h1_rope|...].
    After .T the columns follow this interleaved order.

    The b1 pipeline expects columns grouped as [ALL_NOPE | ALL_ROPE]:
    [h0_nope|h1_nope|...|hN_nope|h0_rope|h1_rope|...|hN_rope].

    Args:
        q_b_proj: HF ``self_attn.q_b_proj.weight`` with shape ``(out_features, in_features)``;
            columns are interleaved per head before ``.T`` inside this function.
        num_heads: Number of attention heads. If None, inferred from the width after transpose.

    Returns:
        Tensor of the same shape with columns reordered to [ALL_NOPE | ALL_ROPE].
    """
    q_b_transposed = q_b_proj.T
    K, N = q_b_transposed.shape
    if num_heads is None:
        num_heads = N // _Q_HEAD_DIM
    heads = q_b_transposed.reshape(K, num_heads, _Q_HEAD_DIM)
    nope = heads[:, :, :_QK_NOPE_HEAD_DIM].reshape(K, -1)
    rope = heads[:, :, _QK_NOPE_HEAD_DIM:].reshape(K, -1)
    return torch.cat([nope, rope], dim=1).contiguous()


def split_kv_b_proj(kv_b_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split HF kv_b_proj (out_features, in_features) into kv_b1 and kv_b2.

    Expects full logical shape (32768, 512) for 4x2 mesh.
    out_features = num_heads * (qk_nope_head_dim + v_head_dim) = num_heads * 256.
    Reshape to (num_heads, 256, 512); first 128 dims are k (b1), last 128 are v (b2).
    Only kv_b2 is transposed for blitz.
    """
    out_features, kv_lora_rank = kv_b_proj.shape
    assert kv_lora_rank == _KV_LORA_RANK
    num_heads = out_features // _KV_B_PROJ_HEAD_DIM
    w = kv_b_proj.reshape(num_heads, _KV_B_PROJ_HEAD_DIM, _KV_LORA_RANK).contiguous()
    kv_b1 = w[:, :_QK_NOPE_HEAD_DIM, :].reshape(-1, _KV_LORA_RANK)
    kv_b2 = w[:, _QK_NOPE_HEAD_DIM:, :].reshape(-1, _KV_LORA_RANK).T.contiguous()
    return kv_b1, kv_b2


def transform_eh_proj(eh_proj_weight_T: torch.Tensor) -> torch.Tensor:
    """Pad to DRAM bank alignment and tile-shuffle. Input: already transposed (K, N)."""
    K, N = eh_proj_weight_T.shape
    assert N % _MTP_NUM_DRAM_BANKS == 0, f"eh_proj N={N} must be divisible by {_MTP_NUM_DRAM_BANKS} DRAM banks"
    n_per_bank = N // _MTP_NUM_DRAM_BANKS
    padded_N = _MTP_NUM_DRAM_BANKS * n_per_bank
    eh_padded = torch.zeros((K, padded_N), dtype=eh_proj_weight_T.dtype)
    eh_padded[:, :N] = eh_proj_weight_T
    return BlitzDecodeWeights._shuffle_dram_tiles(eh_padded, 32, _MTP_NUM_DRAM_BANKS).contiguous()


def mtp_eh_proj_preprocess(raw: dict[str, torch.Tensor], src_key: str, target_name: str) -> dict[str, torch.Tensor]:
    """Preprocess eh_proj for cache: transpose, pad to DRAM bank alignment, tile-shuffle."""
    return {target_name: transform_eh_proj(raw[src_key].T.contiguous())}
