# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 rotary tables.

Ratio-0 layers (sliding window only) rotate at ``ROPE_THETA`` without YaRN; compressed layers rotate
q, window KV, compressed KV and index keys at ``COMPRESS_ROPE_THETA`` with YaRN (``inference/model.py``
``precompute_freqs_cis``). Pairs are adjacent elements, so the tables repeat each frequency twice to
match ``rotary_embedding_llama``'s interleaved layout.
"""

import math

import torch


def inv_freq(config, compressed: bool) -> torch.Tensor:
    """[rope_head_dim / 2] fp32 inverse frequencies of a compressed or ratio-0 layer."""
    dim = config.QK_ROPE_HEAD_DIM
    base = config.COMPRESS_ROPE_THETA if compressed else config.ROPE_THETA
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if not compressed:
        return freqs
    original = config.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS

    def corrected_dim(rotations):
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(corrected_dim(config.ROPE_SCALING_BETA_FAST)), 0)
    high = min(math.ceil(corrected_dim(config.ROPE_SCALING_BETA_SLOW)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
    smooth = 1 - ramp
    return freqs / config.ROPE_SCALING_FACTOR * (1 - smooth) + freqs * smooth


def cos_sin(config, compressed: bool, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [1, 1, len(positions), rope_head_dim] fp32 for integer ``positions``, pair-interleaved."""
    angles = torch.outer(positions.to(torch.float32), inv_freq(config, compressed))
    cos, sin = torch.cos(angles), torch.sin(angles)
    return tuple(t.repeat_interleave(2, dim=-1).view(1, 1, len(positions), -1) for t in (cos, sin))
