# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RoPE utilities for Dots OCR to ensure alignment between HF reference and TTNN.

Dots uses Qwen2-style RoPE. This provides host cos/sin matrices that match
what `DotsTransformer.prepare_inputs_prefill()` expects.
"""

from __future__ import annotations

import warnings

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


def get_rot_mats_hf(seq_len: int, dim: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    HF-style RoPE cos/sin with full head_dim (not half).

    `ttnn.experimental.rotary_embedding` expects `cos_cached`/`sin_cached` shaped like HF:
    `[1, 1, seq_len, head_dim]`, where the first half is repeated in the second half.
    """
    from models.tt_transformers.tt.common import precompute_freqs

    # Match tt_transformers' HF-format precompute (supports rope scaling / types).
    cos_freqs, sin_freqs = precompute_freqs(
        dim,
        seq_len * 2,
        theta=theta,
        scale_factor=None,
        orig_context_len=None,
        rope_type="llama3",
    )
    cos_hf = torch.cat([cos_freqs[:seq_len], cos_freqs[:seq_len]], dim=-1).unsqueeze(0).unsqueeze(0)
    sin_hf = torch.cat([sin_freqs[:seq_len], sin_freqs[:seq_len]], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos_hf, sin_hf


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
    # Precompute only `seq_len` positions (older code used max(seq_len, max_seq_len),
    # which always allocated 8192 rows and wasted RAM on short prompts).
    if seq_len > max_seq_len:
        warnings.warn(
            f"get_rot_mats: seq_len={seq_len} exceeds max_seq_len={max_seq_len}; using seq_len for precompute",
            stacklevel=2,
        )
    end = max(seq_len, 1)
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

    # Prefer Dots/Qwen2 rope_parameters when available (theta/rope_type).
    rope_params = getattr(config, "rope_parameters", None) or {}
    theta = float(rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
    rope_type = rope_params.get("rope_type", "llama3")

    # Scaling parameters (if present) use the same schema as tt_transformers' rope_scaling_model_factory.
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    scale_factor = rope_scaling.get("factor", None)
    orig_context_len = rope_scaling.get("original_max_position_embeddings", None)

    from models.tt_transformers.tt.common import precompute_freqs

    seq_len = input_ids.shape[1]
    cos_freqs, sin_freqs = precompute_freqs(
        head_dim,
        seq_len * 2,
        theta=theta,
        scale_factor=scale_factor,
        orig_context_len=orig_context_len,
        rope_type=rope_type,
    )
    cos_hf = torch.cat([cos_freqs[:seq_len], cos_freqs[:seq_len]], dim=-1).unsqueeze(0).unsqueeze(0)
    sin_hf = torch.cat([sin_freqs[:seq_len], sin_freqs[:seq_len]], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos_hf, sin_hf


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
