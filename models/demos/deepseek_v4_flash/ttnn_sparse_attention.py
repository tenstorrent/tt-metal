# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.cpu_reference import sparse_attention


class TtSparsePrefillAttention(LightweightModule):
    """Single-device DeepSeek V4 Flash sparse attention ABI smoke path.

    Data contract:
    - q: TTNN tensor shaped [batch, 1, seq_len, num_heads * head_dim]
    - compressed_kv: TTNN tensor shaped [batch, 1, cache_len, head_dim]
    - attn_sink: host torch tensor shaped [num_heads]
    - topk_idxs: host torch int tensor shaped [batch, seq_len, topk], with -1 for invalid entries
    - output: TTNN tensor shaped [batch, 1, seq_len, num_heads * head_dim]

    The indexed gather, sink-aware softmax, and weighted reduction currently run on
    host. TTNN does not yet expose a compact primitive for per-token variable
    top-k gather plus the DeepSeek attention-sink softmax in this layout, so this
    module preserves the callable device ABI while the optimized sparse attention
    kernel is still being brought up.
    """

    def __init__(
        self,
        *,
        device,
        num_heads: int,
        head_dim: int,
        softmax_scale: float,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        self.device = device
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.softmax_scale = float(softmax_scale)
        self.dtype = dtype
        self.memory_config = memory_config

    def forward(
        self,
        q,
        compressed_kv,
        *,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
    ):
        q_host = _ttnn_q_to_torch_4d(q, num_heads=self.num_heads, head_dim=self.head_dim)
        kv_host = _ttnn_kv_to_torch_3d(compressed_kv, head_dim=self.head_dim)
        validate_sparse_attention_contract(
            q_host,
            kv_host,
            attn_sink,
            topk_idxs,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        output = sparse_attention(q_host, kv_host, attn_sink, topk_idxs, self.softmax_scale)
        batch_size, seq_len, num_heads, head_dim = output.shape
        output = output.reshape(batch_size, seq_len, num_heads * head_dim).unsqueeze(1).to(torch.bfloat16)
        return ttnn.from_torch(
            output,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )


def validate_sparse_attention_contract(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> None:
    """Validate the host-visible sparse attention ABI before the host fallback path."""

    if q.ndim != 4:
        raise ValueError(f"Expected q shape [batch, seq_len, num_heads, head_dim], got {tuple(q.shape)}")
    if compressed_kv.ndim != 3:
        raise ValueError(f"Expected compressed_kv shape [batch, cache_len, head_dim], got {tuple(compressed_kv.shape)}")
    if attn_sink.ndim != 1:
        raise ValueError(f"Expected attn_sink shape [num_heads], got {tuple(attn_sink.shape)}")
    if topk_idxs.ndim != 3:
        raise ValueError(f"Expected topk_idxs shape [batch, seq_len, topk], got {tuple(topk_idxs.shape)}")
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Expected topk_idxs dtype int32 or int64, got {topk_idxs.dtype}")

    batch_size, seq_len, q_num_heads, q_head_dim = q.shape
    kv_batch_size, cache_len, kv_head_dim = compressed_kv.shape
    if q_num_heads != num_heads or q_head_dim != head_dim:
        raise ValueError(f"Expected q heads/dim {(num_heads, head_dim)}, got {(q_num_heads, q_head_dim)}")
    if kv_batch_size != batch_size:
        raise ValueError(f"Expected compressed_kv batch {batch_size}, got {kv_batch_size}")
    if kv_head_dim != head_dim:
        raise ValueError(f"Expected compressed_kv head_dim {head_dim}, got {kv_head_dim}")
    if tuple(attn_sink.shape) != (num_heads,):
        raise ValueError(f"Expected attn_sink shape {(num_heads,)}, got {tuple(attn_sink.shape)}")
    if tuple(topk_idxs.shape[:2]) != (batch_size, seq_len):
        raise ValueError(f"Expected topk_idxs batch/seq {(batch_size, seq_len)}, got {tuple(topk_idxs.shape[:2])}")
    if torch.any(topk_idxs < -1):
        raise ValueError("topk_idxs values must be -1 or non-negative cache indices")
    if torch.any(topk_idxs >= cache_len):
        raise ValueError(f"topk_idxs values must be < cache_len {cache_len}")


def _ttnn_q_to_torch_4d(q, *, num_heads: int, head_dim: int) -> torch.Tensor:
    torch_q = ttnn.to_torch(q)
    if torch_q.ndim != 4 or torch_q.shape[1] != 1:
        raise ValueError(f"Expected q TTNN shape [batch, 1, seq_len, num_heads * head_dim], got {tuple(torch_q.shape)}")
    expected_hidden = num_heads * head_dim
    if torch_q.shape[-1] != expected_hidden:
        raise ValueError(f"Expected q hidden dim {expected_hidden}, got {torch_q.shape[-1]}")
    batch_size, _, seq_len, _ = torch_q.shape
    return torch_q[:, 0].reshape(batch_size, seq_len, num_heads, head_dim).contiguous()


def _ttnn_kv_to_torch_3d(compressed_kv, *, head_dim: int) -> torch.Tensor:
    torch_kv = ttnn.to_torch(compressed_kv)
    if torch_kv.ndim != 4 or torch_kv.shape[1] != 1:
        raise ValueError(
            f"Expected compressed_kv TTNN shape [batch, 1, cache_len, head_dim], got {tuple(torch_kv.shape)}"
        )
    if torch_kv.shape[-1] != head_dim:
        raise ValueError(f"Expected compressed_kv head_dim {head_dim}, got {torch_kv.shape[-1]}")
    return torch_kv[:, 0].contiguous()
