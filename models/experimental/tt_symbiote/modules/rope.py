# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Rotary Position Embedding (RoPE) implementations for TTNN."""

from typing import Tuple
import torch
import torch.nn as nn

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class TorchRotaryPositionEmbedding(nn.Module):
    """PyTorch implementation of Rotary Position Embedding."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q: The query tensor.
            k: The key tensor.
            cos: The cosine part of the rotary embedding.
            sin: The sine part of the rotary embedding.
            unsqueeze_dim: The dimension along which to unsqueeze cos and sin.

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        # Keep half or full tensor for later concatenation
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        # Apply rotary embeddings
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

        # Concatenate back to full shape
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
        return q_embed, k_embed


class TTNNRotaryPositionEmbedding(TTNNModule):
    """TTNN-accelerated Rotary Position Embedding."""

    def __init__(self):
        """Initialize TTNN RoPE module with minimal setup."""
        super().__init__()
        self._fallback_torch_layer = TorchRotaryPositionEmbedding()

    @classmethod
    def from_torch(cls, layer):
        """Create TTNNRotaryPositionEmbedding with no configuration - learned during forward pass."""
        result = cls()
        result._fallback_torch_layer = layer
        return result

    def forward(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass through RoPE layer.

        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine position embeddings
            sin: Sine position embeddings
            fused_sequence_threshold: Sequence length threshold for using fused operation (default: 128)

        Returns:
            Tuple of (rotated_query, rotated_key)
        """

        if q.layout != ttnn.TILE_LAYOUT:
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if k.layout != ttnn.TILE_LAYOUT:
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if cos.layout != ttnn.TILE_LAYOUT:
            cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if sin.layout != ttnn.TILE_LAYOUT:
            sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if len(sin.shape) == 3:
            sin = ttnn.unsqueeze(sin, dim=0)
        if len(cos.shape) == 3:
            cos = ttnn.unsqueeze(cos, dim=0)
        # Infer configuration from inputs
        batch_size, n_q_heads, seq_len, head_dim = q.shape
        batch_size2, n_k_heads, seq_len2, head_dim2 = k.shape
        assert seq_len == seq_len2, "Query and Key sequence lengths must match."
        assert batch_size == batch_size2, "Query and Key batch sizes must match."
        assert head_dim == head_dim2, "Query and Key head dimensions must match."

        # Store original dimensions
        original_head_dim = head_dim
        original_seq_len = seq_len
        rotary_dim = cos.shape[-1]

        # Handle partial rotary embedding (when cos/sin dim < head_dim)
        if rotary_dim < head_dim:
            # Split q and k into rotary and pass-through portions
            # q/k: [batch, heads, seq, head_dim] -> [batch, heads, seq, rotary_dim] + [batch, heads, seq, head_dim-rotary_dim]
            q_rot = q[:, :, :, :rotary_dim]
            q_pass = q[:, :, :, rotary_dim:]
            k_rot = k[:, :, :, :rotary_dim]
            k_pass = k[:, :, :, rotary_dim:]

            # Pad cos/sin to match tile boundaries if needed
            padded_rotary_dim = rotary_dim
            if rotary_dim % 32 != 0:
                padded_rotary_dim = ((rotary_dim + 31) // 32) * 32
                cos = ttnn.pad(cos, [1, 1, cos.shape[-2], padded_rotary_dim], [0, 0, 0, 0], 0.0)
                sin = ttnn.pad(sin, [1, 1, sin.shape[-2], padded_rotary_dim], [0, 0, 0, 0], 0.0)
                # Also pad q_rot and k_rot
                q_rot = ttnn.pad(q_rot, [batch_size, n_q_heads, seq_len, padded_rotary_dim], [0, 0, 0, 0], 0.0)
                k_rot = ttnn.pad(k_rot, [batch_size2, n_k_heads, seq_len2, padded_rotary_dim], [0, 0, 0, 0], 0.0)

            # Apply rotation to rotary portion only
            q_rot_embedded = ttnn.experimental.rotary_embedding(q_rot, cos, sin)
            k_rot_embedded = ttnn.experimental.rotary_embedding(k_rot, cos, sin)

            # Slice back to original dimensions if padding occurred
            if q_rot_embedded.shape[-2] != seq_len:
                q_rot_embedded = q_rot_embedded[:, :, :seq_len, :]
            if k_rot_embedded.shape[-2] != seq_len:
                k_rot_embedded = k_rot_embedded[:, :, :seq_len, :]
            if padded_rotary_dim != rotary_dim:
                q_rot_embedded = q_rot_embedded[:, :, :, :rotary_dim]
                k_rot_embedded = k_rot_embedded[:, :, :, :rotary_dim]

            # Concatenate rotated and pass-through portions
            q_rotated = ttnn.concat([q_rot_embedded, q_pass], dim=-1)
            k_rotated = ttnn.concat([k_rot_embedded, k_pass], dim=-1)
        else:
            # Full rotary embedding - pad if needed for tile boundaries
            if rotary_dim != head_dim:
                cos = ttnn.pad(cos, [1, 1, cos.shape[-2], head_dim], [0, 0, 0, 0], 0.0)
                sin = ttnn.pad(sin, [1, 1, sin.shape[-2], head_dim], [0, 0, 0, 0], 0.0)

            q_rotated = ttnn.experimental.rotary_embedding(q, cos, sin)
            k_rotated = ttnn.experimental.rotary_embedding(k, cos, sin)

        # Slice back to original dimensions if padding occurred
        if q_rotated.shape[-1] != original_head_dim:
            q_rotated = q_rotated[:, :, :, :original_head_dim]
        if k_rotated.shape[-1] != original_head_dim:
            k_rotated = k_rotated[:, :, :, :original_head_dim]
        if q_rotated.shape[-2] != original_seq_len:
            q_rotated = q_rotated[:, :, :original_seq_len, :]
        if k_rotated.shape[-2] != original_seq_len:
            k_rotated = k_rotated[:, :, :original_seq_len, :]

        return q_rotated, k_rotated
