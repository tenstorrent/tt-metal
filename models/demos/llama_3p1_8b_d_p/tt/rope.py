# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side Llama-3.1 RoPE mathematics and coordinate conversion."""

import math

import torch

from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig


def _require_even_head_dim(head_dim: int) -> None:
    if head_dim % 2:
        raise ValueError(f"RoPE head dimension must be even, got {head_dim}")


def llama3_inv_freq(
    *,
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    theta: float = Llama31_8BConfig.ROPE_THETA,
    factor: float = Llama31_8BConfig.ROPE_SCALING_FACTOR,
    low_freq_factor: float = Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR,
    high_freq_factor: float = Llama31_8BConfig.ROPE_HIGH_FREQ_FACTOR,
    original_max_position_embeddings: int = Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return Llama3-scaled inverse frequencies as a float32 host tensor."""
    _require_even_head_dim(head_dim)

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    wavelength = 2 * math.pi / inv_freq
    low_freq_wavelength = original_max_position_embeddings / low_freq_factor
    high_freq_wavelength = original_max_position_embeddings / high_freq_factor

    scaled_inv_freq = inv_freq / factor
    llama3_freq = torch.where(wavelength > low_freq_wavelength, scaled_inv_freq, inv_freq)
    smooth_factor = (original_max_position_embeddings / wavelength - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (1 - smooth_factor) * scaled_inv_freq + smooth_factor * inv_freq
    medium_freq = (wavelength >= high_freq_wavelength) & (wavelength <= low_freq_wavelength)
    return torch.where(medium_freq, smoothed_inv_freq, llama3_freq)


def build_llama3_cos_sin(
    seq_len: int,
    *,
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    theta: float = Llama31_8BConfig.ROPE_THETA,
    factor: float = Llama31_8BConfig.ROPE_SCALING_FACTOR,
    low_freq_factor: float = Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR,
    high_freq_factor: float = Llama31_8BConfig.ROPE_HIGH_FREQ_FACTOR,
    original_max_position_embeddings: int = Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Meta-interleaved float32 cos/sin tables with shape [1, 1, S, D]."""
    inv_freq = llama3_inv_freq(
        head_dim=head_dim,
        theta=theta,
        factor=factor,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        device=device,
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=inv_freq.device)
    angles = torch.repeat_interleave(torch.outer(positions, inv_freq), 2, dim=-1)
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def hf_to_meta(tensor: torch.Tensor) -> torch.Tensor:
    """Convert final-axis HF half-split coordinates to Meta adjacent pairs."""
    head_dim = tensor.shape[-1]
    _require_even_head_dim(head_dim)
    half = head_dim // 2
    return torch.stack((tensor[..., :half], tensor[..., half:]), dim=-1).flatten(-2)


def meta_to_hf(tensor: torch.Tensor) -> torch.Tensor:
    """Convert final-axis Meta adjacent pairs to HF half-split coordinates."""
    _require_even_head_dim(tensor.shape[-1])
    return torch.cat((tensor[..., 0::2], tensor[..., 1::2]), dim=-1)
