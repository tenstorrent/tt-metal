# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RoPE utilities for Dots OCR to ensure alignment between HF reference and TTNN.

Dots uses Qwen2-style RoPE. This provides host cos/sin matrices that match
what `DotsTransformer.prepare_inputs_prefill()` expects.
"""

from __future__ import annotations

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin matrices for RoPE.

    Returns:
        (cos_matrix, sin_matrix) both with shape [1, 1, end, dim//2]
        This format matches what the TTNN transformer expects.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()  # [end, dim//2]

    # Compute cos and sin
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, end, dim//2]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, end, dim//2]

    return cos, sin


def get_rot_mats(
    seq_len: int,
    dim: int = 128,  # head_dim, typical for Qwen2
    theta: float = 10000.0,
    max_seq_len: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rotation matrices (cos, sin) for a given sequence length.

    This matches the HF Qwen2 RoPE implementation and the format expected
    by `DotsTransformer.prepare_inputs_prefill()`.
    """
    # Ensure we have enough length
    end = max(seq_len, max_seq_len)
    cos_matrix, sin_matrix = precompute_freqs_cis(dim, end, theta)

    # Return slices for the actual sequence length
    return cos_matrix[:, :, :seq_len, :], sin_matrix[:, :, :seq_len, :]


def get_hf_rot_mats_from_model(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract RoPE matrices from HF model to ensure exact alignment.

    For Qwen2-based models like Dots, we can get the rotary embedding
    parameters from the model config.
    """
    config = model.config
    head_dim = getattr(
        config, "head_dim", getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32)
    )

    # Get theta from config or use default
    theta = getattr(config, "rope_theta", 10000.0)

    seq_len = input_ids.shape[1]
    return get_rot_mats(seq_len=seq_len, dim=head_dim, theta=theta)


class Qwen2RopeHelper:
    """
    Helper to ensure RoPE alignment between HF reference and TTNN implementation.
    """

    def __init__(self, head_dim: int = 128, max_seq_len: int = 8192, theta: float = 10000.0):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

    def get_rot_mats(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos, sin matrices for given sequence length."""
        return get_rot_mats(
            seq_len=seq_len,
            dim=self.head_dim,
            theta=self.theta,
            max_seq_len=self.max_seq_len,
        )

    def get_rot_mats_for_batch(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get rot mats for a batch of input_ids."""
        seq_len = input_ids.shape[1]
        return self.get_rot_mats(seq_len)
