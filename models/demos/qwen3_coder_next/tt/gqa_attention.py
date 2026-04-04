# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full GQA attention for Qwen3-Coder-Next (every 4th layer).

Standard Grouped-Query Attention with:
- 16 query heads, 2 KV heads (8:1 GQA ratio)
- head_dim = 256
- Partial RoPE (25% of dims rotated)
- Supports both prefill (parallel) and decode (sequential) modes

Used in layers where layer_idx % full_attention_interval == (full_attention_interval - 1),
i.e., layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47.

Reference: Standard GQA from tt_transformers/tt/attention.py adapted for
Qwen3-Coder-Next's extreme 8:1 ratio and partial RoPE.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.rope import apply_partial_rope_torch


class GQAAttention(nn.Module):
    """Grouped-Query Attention with partial RoPE.

    PyTorch reference implementation for correctness validation.
    """

    def __init__(self, config: Qwen3CoderNextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # 16
        self.num_kv_heads = config.num_key_value_heads  # 2
        self.head_dim = config.head_dim  # 256
        self.num_groups = self.num_heads // self.num_kv_heads  # 8
        self.partial_rotary_factor = config.partial_rotary_factor

        self.scale = self.head_dim**-0.5

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for GQA attention.

        Args:
            hidden_states: (batch, seq_len, hidden_size).
            cos, sin: RoPE frequencies (seq_len, rotary_dim/2).
            attention_mask: Optional causal mask (batch, 1, seq_len, kv_seq_len).
            kv_cache: Optional tuple of (cached_keys, cached_values) for decode mode.
            position_ids: Optional position indices for RoPE.

        Returns:
            Tuple of (output, updated_kv_cache).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply partial RoPE
        if cos is not None and sin is not None:
            q = apply_partial_rope_torch(q, cos, sin, self.partial_rotary_factor)
            k = apply_partial_rope_torch(k, cos, sin, self.partial_rotary_factor)

        # Update KV cache if provided (decode mode)
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
        new_kv_cache = (k, v)

        # Repeat KV heads for GQA
        k = k.repeat_interleave(self.num_groups, dim=2)  # (B, kv_seq, num_heads, head_dim)
        v = v.repeat_interleave(self.num_groups, dim=2)

        # Transpose to (B, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to (B, S, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, new_kv_cache

    @staticmethod
    def make_causal_mask(seq_len: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Create a causal attention mask.

        Args:
            seq_len: Sequence length.
            dtype: Output dtype.

        Returns:
            Causal mask (1, 1, seq_len, seq_len) with -inf for future positions.
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)
