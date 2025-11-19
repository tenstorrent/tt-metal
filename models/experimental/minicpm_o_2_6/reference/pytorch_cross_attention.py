# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation of Cross-Attention layer for MiniCPM-o-2_6.

Simplified implementation for PCC validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PyTorchCrossAttention(nn.Module):
    """
    PyTorch reference implementation of Cross-Attention for PCC validation.

    This performs cross-attention between LLM hidden states and multimodal embeddings.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        multimodal_dim: int = 3584,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.multimodal_dim = multimodal_dim

        self.head_dim = hidden_size // num_attention_heads

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(multimodal_dim, num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(multimodal_dim, num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # RMS norms (following Qwen2)
        self.q_norm = nn.RMSNorm(self.head_dim)  # Applied per head
        self.k_norm = nn.RMSNorm(self.head_dim)  # Applied per head after projection

    def forward(
        self,
        hidden_states: torch.Tensor,
        multimodal_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            multimodal_embeds: [batch_size, multimodal_seq_len, multimodal_dim]
            attention_mask: Optional [batch_size, seq_len, multimodal_seq_len]

        Returns:
            torch.Tensor: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        _, multimodal_seq_len, _ = multimodal_embeds.shape

        # Query projection
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        q = self.q_norm(q)  # Apply RMS norm per head (after reshaping)

        # Key/Value projections
        k = self.k_proj(multimodal_embeds)
        k = k.view(batch_size, multimodal_seq_len, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(k)  # Apply RMS norm per head

        v = self.v_proj(multimodal_embeds)
        v = v.view(batch_size, multimodal_seq_len, self.num_key_value_heads, self.head_dim)

        # Transpose for attention: [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle GQA
        if self.num_key_value_heads < self.num_attention_heads:
            repeats = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output
