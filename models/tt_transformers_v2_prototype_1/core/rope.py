# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Rotary Position Embeddings (RoPE) module"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import ttnn


@dataclass
class RoPEConfig:
    """Configuration for Rotary Position Embeddings"""

    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000.0
    scaling_factor: Optional[float] = None
    rope_type: str = "default"  # Options: "default", "linear", "dynamic"
    partial_rope_factor: float = 1.0  # Apply RoPE to partial dimensions


class RoPE(torch.nn.Module):
    """
    Rotary Position Embeddings module.

    Implements RoPE as described in the RoFormer paper.
    Supports various scaling methods for extending context length.
    """

    def __init__(
        self,
        config: RoPEConfig,
        device: ttnn.Device,
    ):
        super().__init__()
        self.config = config
        self.device = device

        # Precompute frequency tensor
        self._precompute_freqs()

        # Cache for cos/sin values
        self.cos_cached = None
        self.sin_cached = None
        self.max_seq_len_cached = 0

    def _precompute_freqs(self):
        """Precompute the frequency tensor for RoPE"""
        dim = int(self.config.dim * self.config.partial_rope_factor)
        freqs = 1.0 / (self.config.base ** (torch.arange(0, dim, 2).float() / dim))

        # Apply scaling if configured
        if self.config.scaling_factor is not None:
            freqs = freqs / self.config.scaling_factor

        self.inv_freqs = freqs

    def _update_cache(self, seq_len: int):
        """Update the cos/sin cache if needed"""
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            # Create position indices
            t = torch.arange(seq_len, dtype=torch.float32)

            # Apply rope scaling if needed
            if self.config.rope_type == "linear":
                t = t / self.config.scaling_factor
            elif self.config.rope_type == "dynamic":
                # Dynamic scaling based on position
                scale = seq_len / self.config.max_position_embeddings
                t = t / scale

            # Compute frequencies
            freqs = torch.outer(t, self.inv_freqs)

            # Compute cos and sin
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)

            # Cache the values
            self.cos_cached = ttnn.from_torch(cos, device=self.device, layout=ttnn.TILE_LAYOUT)
            self.sin_cached = ttnn.from_torch(sin, device=self.device, layout=ttnn.TILE_LAYOUT)

    def forward(
        self,
        positions: ttnn.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Get cos and sin values for given positions.

        Args:
            positions: Position indices
            seq_len: Maximum sequence length to cache

        Returns:
            cos: Cosine values for rotary embeddings
            sin: Sine values for rotary embeddings
        """
        if seq_len is None:
            seq_len = int(ttnn.max(positions).item()) + 1

        # Update cache if needed
        self._update_cache(seq_len)

        # Get cos/sin values for the positions
        cos = ttnn.embedding(positions, self.cos_cached)
        sin = ttnn.embedding(positions, self.sin_cached)

        return cos, sin

    @staticmethod
    def apply_rotary_embeddings(
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        position_ids: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape [batch, heads, seq_len, head_dim]
            k: Key tensor of shape [batch, heads, seq_len, head_dim]
            cos: Cosine values from RoPE
            sin: Sine values from RoPE
            position_ids: Optional custom position IDs

        Returns:
            q_rotated: Query with rotary embeddings applied
            k_rotated: Key with rotary embeddings applied
        """
        # Get dimensions
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Reshape cos/sin for broadcasting
        cos = ttnn.reshape(cos, [1, 1, seq_len, head_dim // 2])
        sin = ttnn.reshape(sin, [1, 1, seq_len, head_dim // 2])

        # Split q and k into two halves
        q_r, q_i = ttnn.split(q, head_dim // 2, dim=-1)
        k_r, k_i = ttnn.split(k, head_dim // 2, dim=-1)

        # Apply rotation using complex number formula
        # (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
        q_r_rot = q_r * cos - q_i * sin
        q_i_rot = q_r * sin + q_i * cos
        k_r_rot = k_r * cos - k_i * sin
        k_i_rot = k_r * sin + k_i * cos

        # Concatenate back
        q_rotated = ttnn.concat([q_r_rot, q_i_rot], dim=-1)
        k_rotated = ttnn.concat([k_r_rot, k_i_rot], dim=-1)

        return q_rotated, k_rotated


class VisionRoPE2D(torch.nn.Module):
    """
    2D Rotary Position Embeddings for vision transformers.

    Extends RoPE to 2D positions for image patches.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 1024,
        base: float = 10000.0,
        device: ttnn.Device = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device

        # We use half dim for each spatial dimension (h, w)
        self.rope_h = RoPE(
            RoPEConfig(
                dim=dim // 2,
                max_position_embeddings=max_position_embeddings,
                base=base,
            ),
            device=device,
        )
        self.rope_w = RoPE(
            RoPEConfig(
                dim=dim // 2,
                max_position_embeddings=max_position_embeddings,
                base=base,
            ),
            device=device,
        )

    def forward(
        self,
        height: int,
        width: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Get 2D rotary embeddings for image patches.

        Args:
            height: Height in patches
            width: Width in patches

        Returns:
            cos: Cosine values for 2D rotary embeddings
            sin: Sine values for 2D rotary embeddings
        """
        # Create 2D position grid
        h_pos = ttnn.arange(height, device=self.device)
        w_pos = ttnn.arange(width, device=self.device)

        # Get 1D rotary embeddings for each dimension
        cos_h, sin_h = self.rope_h(h_pos, height)
        cos_w, sin_w = self.rope_w(w_pos, width)

        # Combine into 2D embeddings
        # This is a simplified implementation - actual implementation
        # would properly handle the 2D grid combination
        cos_h = ttnn.broadcast_to(cos_h, [height, width, self.dim // 2])
        sin_h = ttnn.broadcast_to(sin_h, [height, width, self.dim // 2])
        cos_w = ttnn.broadcast_to(cos_w, [height, width, self.dim // 2])
        sin_w = ttnn.broadcast_to(sin_w, [height, width, self.dim // 2])

        # Concatenate height and width embeddings
        cos_2d = ttnn.concat([cos_h, cos_w], dim=-1)
        sin_2d = ttnn.concat([sin_h, sin_w], dim=-1)

        return cos_2d, sin_2d
