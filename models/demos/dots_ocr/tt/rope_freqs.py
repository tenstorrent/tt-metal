# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import torch


def _compute_llama3_parameters(freqs: torch.Tensor, scale_factor: float, orig_context_len: int) -> torch.Tensor:
    # Llama-3.x scaling (matches tt_transformers logic)
    low_freq_factor = 1.0
    high_freq_factor = 4.0

    low_freq_wavelen = orig_context_len / low_freq_factor
    high_freq_wavelen = orig_context_len / high_freq_factor

    out = []
    for freq in freqs:
        wavelen = 2 * math.pi / float(freq)
        if wavelen < high_freq_wavelen:
            out.append(freq)
        elif wavelen > low_freq_wavelen:
            out.append(freq / scale_factor)
        else:
            # Smooth interpolation between scaled/unscaled region
            smooth = (orig_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            out.append((1.0 - smooth) * (freq / scale_factor) + smooth * freq)
    return torch.tensor(out, dtype=freqs.dtype, device=freqs.device)


def _apply_scaling(
    freqs: torch.Tensor,
    *,
    scale_factor: Optional[float],
    orig_context_len: Optional[int],
    rope_type: str,
) -> torch.Tensor:
    if scale_factor is None:
        return freqs
    if orig_context_len is None:
        # Some configs provide scale without explicit original context length;
        # in that case, fall back to unscaled behavior.
        return freqs

    if rope_type in ("default", "mrope"):
        return freqs
    if rope_type == "linear":
        return freqs / float(scale_factor)
    if rope_type == "llama3":
        return _compute_llama3_parameters(freqs, float(scale_factor), int(orig_context_len))

    # Unknown rope_type: safest is no scaling rather than guessing.
    return freqs


def precompute_freqs(
    dim: int,
    end: int,
    *,
    theta: float,
    scale_factor: Optional[float],
    orig_context_len: Optional[int],
    rope_type: str = "llama3",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dots OCR-local RoPE frequency precompute.

    Returns:
        (cos, sin) shaped [end, dim//2], matching the subset used in Dots OCR
        (and matching the contract relied on by Dots' padding logic).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = _apply_scaling(freqs, scale_factor=scale_factor, orig_context_len=orig_context_len, rope_type=rope_type)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)
