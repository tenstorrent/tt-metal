# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Partial Rotary Position Embeddings for Qwen3-Coder-Next.

Unlike standard RoPE which rotates all head dimensions, Qwen3-Coder-Next
uses partial_rotary_factor=0.25, meaning only 25% of head_dim (64 of 256)
dimensions receive rotary embeddings. The remaining 75% pass through unchanged.

Reference: HuggingFace Qwen3NextForCausalLM RoPE implementation.
"""

from typing import Optional, Tuple

import torch

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig


def precompute_freqs(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float = 5000000.0,
    partial_rotary_factor: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine frequencies for partial RoPE.

    Only computes frequencies for the rotary portion of head_dim.

    Args:
        head_dim: Full head dimension (e.g., 256).
        max_seq_len: Maximum sequence length to precompute.
        rope_theta: RoPE theta parameter.
        partial_rotary_factor: Fraction of head_dim to rotate (0.25).

    Returns:
        Tuple of (cos, sin) tensors, each of shape (max_seq_len, rotary_dim).
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    assert rotary_dim % 2 == 0, f"rotary_dim must be even, got {rotary_dim}"

    # Compute inverse frequencies for the rotary dimensions
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))

    # Compute position-dependent frequencies
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, rotary_dim/2)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return cos, sin


def apply_partial_rope_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    partial_rotary_factor: float = 0.25,
) -> torch.Tensor:
    """Apply partial RoPE in PyTorch (for reference/testing).

    Args:
        x: Input tensor (batch, seq_len, num_heads, head_dim).
        cos: Cosine frequencies (seq_len, rotary_dim/2).
        sin: Sine frequencies (seq_len, rotary_dim/2).
        partial_rotary_factor: Fraction of head_dim to rotate.

    Returns:
        Tensor with partial rotary embeddings applied.
    """
    head_dim = x.shape[-1]
    rotary_dim = int(head_dim * partial_rotary_factor)

    # Split into rotary and pass-through portions
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Split rotary portion into two halves for rotation
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]

    # Reshape cos/sin for broadcasting: (1, seq_len, 1, rotary_dim/2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    x_rot_out = torch.cat(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin],
        dim=-1,
    )

    # Concatenate rotated and pass-through portions
    return torch.cat([x_rot_out, x_pass], dim=-1)


def freqs_to_rotation_matrix(
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """Convert frequency tensors to rotation matrices for TTNN.

    Creates a block-diagonal rotation matrix where:
    - Top-left block is the standard RoPE rotation (rotary_dim x rotary_dim)
    - Bottom-right block is identity (non_rotary_dim x non_rotary_dim)

    Args:
        cos: Cosine frequencies (seq_len, rotary_dim/2).
        sin: Sine frequencies (seq_len, rotary_dim/2).
        rotary_dim: Number of rotary dimensions.
        head_dim: Total head dimension.

    Returns:
        Rotation matrix (seq_len, head_dim, head_dim).
    """
    seq_len = cos.shape[0]
    non_rotary_dim = head_dim - rotary_dim
    half_rot = rotary_dim // 2

    # Build 2x2 rotation blocks for each pair of rotary dims
    rot_matrix = torch.zeros(seq_len, head_dim, head_dim)

    for i in range(half_rot):
        # Each pair (2i, 2i+1) gets a rotation block
        rot_matrix[:, i, i] = cos[:, i]
        rot_matrix[:, i, i + half_rot] = -sin[:, i]
        rot_matrix[:, i + half_rot, i] = sin[:, i]
        rot_matrix[:, i + half_rot, i + half_rot] = cos[:, i]

    # Identity block for non-rotary dimensions
    for i in range(non_rotary_dim):
        rot_matrix[:, rotary_dim + i, rotary_dim + i] = 1.0

    return rot_matrix


class PartialRoPE:
    """Partial Rotary Position Embedding manager.

    Precomputes and caches RoPE frequencies, provides methods
    to apply partial rotation to query and key tensors.
    """

    def __init__(self, config: Qwen3CoderNextConfig, max_seq_len: int = 8192):
        self.config = config
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.non_rotary_dim = config.non_rotary_dim
        self.partial_rotary_factor = config.partial_rotary_factor
        self.max_seq_len = max_seq_len

        # Precompute frequencies
        self.cos, self.sin = precompute_freqs(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            rope_theta=config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
        )

    def get_cos_sin(
        self, seq_len: int, position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin for a given sequence length or position IDs.

        Args:
            seq_len: Sequence length (used if position_ids is None).
            position_ids: Optional explicit position IDs (batch, seq_len).

        Returns:
            Tuple of (cos, sin) sliced/gathered for the positions.
        """
        if position_ids is not None:
            # Gather specific positions
            cos = self.cos[position_ids]  # (batch, seq_len, rotary_dim/2)
            sin = self.sin[position_ids]
        else:
            cos = self.cos[:seq_len]  # (seq_len, rotary_dim/2)
            sin = self.sin[:seq_len]
        return cos, sin

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply partial RoPE to query and key tensors.

        Args:
            q: Query tensor (batch, seq_len, num_heads, head_dim).
            k: Key tensor (batch, seq_len, num_kv_heads, head_dim).
            cos: Cosine frequencies.
            sin: Sine frequencies.

        Returns:
            Tuple of rotated (q, k) tensors.
        """
        q_rot = apply_partial_rope_torch(q, cos, sin, self.partial_rotary_factor)
        k_rot = apply_partial_rope_torch(k, cos, sin, self.partial_rotary_factor)
        return q_rot, k_rot
