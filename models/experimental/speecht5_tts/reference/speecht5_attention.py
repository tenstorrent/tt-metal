# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Attention mechanisms for SpeechT5 model.
Adapted from transformers.models.speecht5.modeling_speecht5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpeechT5Attention(nn.Module):
    """
    Multi-headed attention module for SpeechT5.
    Supports both self-attention and cross-attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = hidden_size // num_heads
        self.is_decoder = is_decoder

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        """Reshape tensor for multi-head attention"""
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            key_value_states: [batch, kv_seq_len, hidden_size] for cross-attention
            attention_mask: [batch, 1, seq_len, kv_seq_len]
            position_bias: [batch, num_heads, seq_len, kv_seq_len] or [1, num_heads, seq_len, kv_seq_len]
            past_key_value: cached key and value tensors
            output_attentions: whether to return attention weights

        Returns:
            attn_output: [batch, seq_len, hidden_size]
            attn_weights: [batch, num_heads, seq_len, kv_seq_len] if output_attentions
            present_key_value: cached key and value for next step
        """
        # is_cross_attention determines if we're doing cross-attention
        is_cross_attention = key_value_states is not None

        batch_size, seq_len, _ = hidden_states.size()

        # Get query projection
        query_states = self.q_proj(hidden_states) * self.scaling

        # Get key/value projections
        if is_cross_attention and past_key_value is not None:
            # Reuse cached k, v for cross-attention
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # Cross-attention: keys and values come from encoder
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states), -1, batch_size)
        elif past_key_value is not None:
            # Self-attention with past: concatenate with cached keys/values
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states), -1, batch_size)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # Standard self-attention
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states), -1, batch_size)

        # Cache key/value for next step if this is decoder
        if self.is_decoder:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        # Reshape query
        query_states = self._shape(query_states, seq_len, batch_size)

        # Compute attention scores
        kv_seq_len = key_states.size(2)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Add position bias if provided (for relative positional encoding)
        if position_bias is not None:
            attn_weights = attn_weights + position_bias

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply dropout
        if self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Reshape back to [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Final output projection
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if self.is_decoder:
            outputs += (present_key_value,)

        return outputs
