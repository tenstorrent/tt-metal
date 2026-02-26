# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Rotary Position Embedding (RoPE) for Molmo2 Text Model.

Implements RoPE with high theta (1,000,000) for long context support.
Precomputes cos/sin tables for efficient position encoding.
"""

from typing import Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TextRotaryEmbedding(LightweightModule):
    """
    Rotary Position Embedding for Molmo2.

    Uses high theta (1M) for extended context length support.
    Precomputes cos/sin values for all positions up to max_seq_len.
    """

    def __init__(
        self,
        mesh_device,
        head_dim: int = 128,
        max_seq_len: int = 8192,
        theta: float = 1000000.0,
        dtype=ttnn.bfloat16,
    ):
        """
        Initialize RotaryEmbedding.

        Args:
            mesh_device: TTNN mesh device or single device
            head_dim: Dimension per attention head (128)
            max_seq_len: Maximum sequence length (8192)
            theta: Base for frequency computation (1,000,000)
            dtype: Data type for embeddings
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.dtype = dtype

        # Precompute cos/sin tables
        self._precompute_freqs(mesh_device)

    def _precompute_freqs(self, device):
        """Precompute cos/sin frequency tables."""
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))

        # Compute position indices
        t = torch.arange(self.max_seq_len, dtype=torch.float32)

        # Compute freqs: [max_seq_len, head_dim // 2]
        freqs = torch.outer(t, inv_freq)

        # Compute cos/sin with interleaved pattern for apply_rotary
        # Shape: [max_seq_len, head_dim]
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)

        # Reshape for broadcast: [1, 1, max_seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

        # Store as TTNN tensors
        self.cos_cached = ttnn.as_tensor(
            cos,
            dtype=self.dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.sin_cached = ttnn.as_tensor(
            sin,
            dtype=self.dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def get_cos_sin(
        self,
        seq_len: int,
        start_pos: int = 0,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Get cos/sin values for a sequence.

        Args:
            seq_len: Sequence length
            start_pos: Starting position for KV cache scenarios

        Returns:
            Tuple of (cos, sin) tensors of shape [1, 1, seq_len, head_dim]
        """
        # Slice precomputed values to correct seq_len
        # Convert to torch, slice, convert back
        cos_torch = ttnn.to_torch(self.cos_cached)
        sin_torch = ttnn.to_torch(self.sin_cached)

        # Slice: [1, 1, max_seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        cos_sliced = cos_torch[:, :, start_pos : start_pos + seq_len, :]
        sin_sliced = sin_torch[:, :, start_pos : start_pos + seq_len, :]

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        cos = ttnn.from_torch(
            cos_sliced,
            dtype=self.dtype,
            device=self.mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.from_torch(
            sin_sliced,
            dtype=self.dtype,
            device=self.mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return cos, sin


def apply_rotary_emb(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
) -> ttnn.Tensor:
    """
    Apply rotary embeddings to input tensor.

    Uses the formula:
        x_out = x * cos + rotate_half(x) * sin

    Where rotate_half swaps pairs and negates the first element of each pair.

    Args:
        x: Input tensor of shape [batch, num_heads, seq_len, head_dim]
        cos: Cosine values of shape [1, 1, seq_len, head_dim]
        sin: Sine values of shape [1, 1, seq_len, head_dim]

    Returns:
        Tensor with rotary embedding applied
    """
    # TTNN has built-in rotary embedding support
    # For now, use the manual formula
    # x_rotated = rotate_half(x)
    # output = x * cos + x_rotated * sin

    # Use TTNN's rotary embedding if available
    # Otherwise implement manually
    return ttnn.transformer.apply_rotary_emb(x, cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input (PyTorch reference).

    For input [..., head_dim], splits into two halves and rotates:
        [x1, x2] -> [-x2, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
