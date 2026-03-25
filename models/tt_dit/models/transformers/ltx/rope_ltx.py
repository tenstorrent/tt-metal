# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 Rotary Position Embeddings for tt_dit.

Computes cos/sin on CPU using LTX-2's fractional position encoding with custom
frequency grids, then converts to ttnn tensors for use with
ttnn.experimental.rotary_embedding_llama.

Reference: LTX-2/packages/ltx-core/src/ltx_core/model/transformer/rope.py
"""

from __future__ import annotations

import functools
import math
from enum import Enum

import numpy as np
import torch


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"


@functools.lru_cache(maxsize=5)
def generate_freq_grid(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> torch.Tensor:
    """
    Generate frequency grid for LTX-2 RoPE.

    Uses a power-law spacing: theta^linspace(log(1)/log(theta), 1, D) * pi/2
    where D = inner_dim // (2 * max_pos_count).
    """
    theta = positional_embedding_theta
    n_elem = 2 * positional_embedding_max_pos_count

    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(1.0) / np.log(theta),
            np.log(theta) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


def get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int]) -> torch.Tensor:
    """Normalize position indices to [0, 1] by dividing by max_pos per dimension.

    Args:
        indices_grid: (B, n_dims, N) position indices after averaging start/end
        max_pos: max position per dimension
    """
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(
        max_pos
    ), f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    return torch.stack([indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)], dim=-1)


def generate_freqs(
    indices: torch.Tensor,
    indices_grid: torch.Tensor,
    max_pos: list[int],
    use_middle_indices_grid: bool,
) -> torch.Tensor:
    """
    Compute raw frequencies from position indices and frequency grid.

    freqs = indices * (fractional_positions * 2 - 1)
    This maps fractional positions from [0,1] to [-1,1] before multiplying by freq indices.
    """
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)
    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def interleaved_freqs_cis(freqs: torch.Tensor, pad_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cos/sin with interleaved repeat for rotary embedding.

    Each frequency value is duplicated: [f0, f0, f1, f1, ...] to match
    the interleaved (x0, x1) pair structure expected by rotary_embedding_llama.
    """
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq


def split_freqs_cis(freqs: torch.Tensor, pad_size: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cos/sin with split layout for rotary_embedding_llama.

    The llama kernel expects (B, H, N, head_dim) and applies rotation to pairs (x[i], x[i+D/2]).
    We repeat the unique frequencies: [f0,f1,...,f_{d/2}, f0,f1,...,f_{d/2}] so both halves
    are rotated correctly.

    Output shape: (B, num_heads, T, head_dim) where head_dim = 2 * D_half.
    """
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    b, t = cos_freq.shape[0], cos_freq.shape[1]
    # Reshape to per-head: (B, T, H, D_half) -> (B, H, T, D_half)
    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1).swapaxes(1, 2)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1).swapaxes(1, 2)
    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype = torch.float32,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin for LTX-2 RoPE.

    Args:
        indices_grid: (seq_len, n_pos_dims) or (B, seq_len, n_pos_dims) position indices
        dim: Hidden dimension (e.g. 4096 for LTX-2)
        out_dtype: Output dtype
        theta: Frequency base (default 10000)
        max_pos: Max position per dimension [temporal, height, width]
        use_middle_indices_grid: If True, average start/end indices from 4D grids
        num_attention_heads: Number of attention heads (for split mode reshape)
        rope_type: INTERLEAVED or SPLIT

    Returns:
        (cos_freq, sin_freq) tuple. Shape depends on rope_type:
        - INTERLEAVED: (B, seq_len, dim) with interleaved duplicated values
        - SPLIT: (B, num_heads, seq_len, dim // (2 * num_heads))
    """
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    # n_pos_dims is the number of position dimensions (e.g. 3 for temporal + H + W)
    # For 4D grids (B, n_dims, N, 2), n_pos_dims is shape[1]; for 3D (B, N, n_dims), it's shape[-1]
    if use_middle_indices_grid and len(indices_grid.shape) == 4:
        n_pos_dims = indices_grid.shape[1]  # (B, n_dims, N, 2)
    else:
        n_pos_dims = indices_grid.shape[-1]
    indices = generate_freq_grid(theta, n_pos_dims, dim)
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        n_elem = 2 * n_pos_dims
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)

    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> torch.Tensor:
    """
    Apply LTX-2 rotary embedding to input tensor (PyTorch reference, for testing).

    Args:
        input_tensor: (B, seq_len, num_heads, head_dim) or (B, num_heads, seq_len, head_dim)
        freqs_cis: (cos_freq, sin_freq) from precompute_freqs_cis
        rope_type: INTERLEAVED or SPLIT
    """
    cos_freqs, sin_freqs = freqs_cis
    if rope_type == LTXRopeType.INTERLEAVED:
        return _apply_interleaved_rotary_emb(input_tensor, cos_freqs, sin_freqs)
    elif rope_type == LTXRopeType.SPLIT:
        return _apply_split_rotary_emb(input_tensor, cos_freqs, sin_freqs)
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")


def _apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    """Apply interleaved RoPE: pairs (x0, x1) rotated by (cos, sin)."""
    t_dup = input_tensor.unflatten(-1, (-1, 2))
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = t_dup.flatten(-2)
    return input_tensor * cos_freqs + input_tensor_rot * sin_freqs


def _apply_split_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    """Apply split RoPE: first-half and second-half rotated independently."""
    needs_reshape = False
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    split_input = input_tensor.unflatten(-1, (2, -1))
    first_half = split_input[..., :1, :]
    second_half = split_input[..., 1:, :]

    output = split_input * cos_freqs.unsqueeze(-2)
    output[..., :1, :] -= sin_freqs.unsqueeze(-2) * second_half
    output[..., 1:, :] += sin_freqs.unsqueeze(-2) * first_half

    output = output.flatten(-2)
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)
    return output
