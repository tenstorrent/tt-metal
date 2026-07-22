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

import ttnn

from ....parallel.config import DiTParallelConfig
from ....utils.patchifiers import (
    AudioLatentShape,
    VideoLatentShape,
    audio_get_patch_grid_bounds,
    get_pixel_coords,
    video_get_patch_grid_bounds,
)
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"


@functools.lru_cache(maxsize=5)
def generate_freq_grid(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> torch.Tensor:
    """Power-law frequency grid for LTX-2 RoPE: theta^linspace(...) * pi/2.

    Kept fp32, not HF's fp64: upgrading the grid to fp64 measurably hurt TT-vs-HF
    audio correlation because the downstream residual stream is bf16. Keep fp32
    unless the whole RoPE + residual path moves to fp32.
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
    else:
        # 3D (B, N, n_dims) → transpose to (B, n_dims, N) for get_fractional_positions
        indices_grid = indices_grid.transpose(1, 2)

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)
    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def reshape_interleaved_to_bhnd(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshape interleaved (B, N, dim) to (B, num_heads, N, head_dim) for rotary_embedding_llama."""
    B, N, dim = t.shape
    head_dim = dim // num_heads
    return t.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


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
    """Precompute (cos, sin) for LTX-2 RoPE.

    Output shape depends on rope_type:
    - INTERLEAVED: (B, seq_len, dim), each freq duplicated for the (x0, x1) pairs.
    - SPLIT: (B, num_heads, seq_len, dim // (2 * num_heads)).
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


# =============================================================================
# Device-side RoPE builders (INTERLEAVED cos/sin, sharded onto the DiT mesh) for
# ttnn.experimental.rotary_embedding_llama. All positions stay fp32 — bf16 introduced
# catastrophic phase error in high-frequency RoPE channels (randomizing the top half of
# head_dim), so only the final ttnn tensors are cast to bf16.
# =============================================================================


def pad_video_rope_sp(
    cos_freq: torch.Tensor, sin_freq: torch.Tensor, sp_factor: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad video RoPE cos/sin on dim=2 to the SP boundary (``ttnn.TILE_SIZE * sp_factor``).

    Padded slots use cos=1, sin=0 (identity rotation); SDPA still masks them via ``logical_n``.
    Same convention as the audio RoPE padding in ``prepare_audio_rope`` / ``prepare_av_cross_pe``.
    """
    video_N_real = cos_freq.shape[2]
    divisor = ttnn.TILE_SIZE * sp_factor
    video_N = ((video_N_real + divisor - 1) // divisor) * divisor
    if video_N == video_N_real:
        return cos_freq, sin_freq
    pad = video_N - video_N_real
    H = cos_freq.shape[1]
    d_half = cos_freq.shape[-1]
    cos_pad = torch.ones(1, H, pad, d_half, dtype=cos_freq.dtype)
    sin_pad = torch.zeros(1, H, pad, d_half, dtype=sin_freq.dtype)
    cos_freq = torch.cat([cos_freq, cos_pad], dim=2)
    sin_freq = torch.cat([sin_freq, sin_pad], dim=2)
    return cos_freq, sin_freq


def prepare_video_rope(
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    inner_dim: int,
    num_attention_heads: int,
    theta: float,
    max_pos: list[int],
    mesh_device: ttnn.MeshDevice,
    parallel_config: DiTParallelConfig,
    fps: float = 24.0,
    anchor_frames: list[int] | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Compute video RoPE in INTERLEAVED layout, SP×TP sharded onto the mesh.

    ``anchor_frames`` appends, at the tail, one h×w block of positions per listed latent frame,
    reusing that frame's own (temporal+spatial) coordinates. Append-token interior keyframes ride
    there so the anchor carries the exact RoPE phase of the free grid frame it conditions."""
    v_shape = VideoLatentShape(batch=1, channels=128, frames=latent_frames, height=latent_height, width=latent_width)
    v_coords = video_get_patch_grid_bounds(v_shape)
    v_positions = get_pixel_coords(v_coords, scale_factors=(8, 32, 32), causal_fix=True).float()
    v_positions[:, 0, ...] = v_positions[:, 0, ...] / fps
    if anchor_frames:
        if any(f < 0 or f >= latent_frames for f in anchor_frames):
            raise ValueError(f"anchor_frames {anchor_frames} out of range [0, {latent_frames})")
        hw = latent_height * latent_width
        v_positions = torch.cat(
            [v_positions] + [v_positions[:, :, f * hw : (f + 1) * hw, :] for f in anchor_frames], dim=2
        )

    cos_freq, sin_freq = precompute_freqs_cis(
        v_positions,
        dim=inner_dim,
        out_dtype=torch.float32,
        theta=theta,
        max_pos=max_pos,
        use_middle_indices_grid=True,
        num_attention_heads=num_attention_heads,
        rope_type=LTXRopeType.INTERLEAVED,
    )  # (1, N, dim)

    cos_freq = reshape_interleaved_to_bhnd(cos_freq, num_attention_heads)
    sin_freq = reshape_interleaved_to_bhnd(sin_freq, num_attention_heads)

    # Pad seq dim to ttnn.TILE_SIZE * sp_factor; padded slots use cos=1, sin=0 (identity).
    cos_freq, sin_freq = pad_video_rope_sp(cos_freq, sin_freq, parallel_config.sequence_parallel.factor)

    sp_axis = parallel_config.sequence_parallel.mesh_axis
    tp_axis = parallel_config.tensor_parallel.mesh_axis
    tt_cos = bf16_tensor_2dshard(cos_freq, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_freq, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    return tt_cos, tt_sin


def prepare_audio_rope(
    audio_N: int,
    audio_N_real: int,
    *,
    theta: float,
    mesh_device: ttnn.MeshDevice,
    parallel_config: DiTParallelConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Compute audio RoPE in INTERLEAVED layout, SP×TP sharded onto the mesh."""
    a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N_real, mel_bins=16)
    a_positions = audio_get_patch_grid_bounds(a_shape).float()  # (1, 1, N, 2)

    a_cos, a_sin = precompute_freqs_cis(
        a_positions,
        dim=2048,
        out_dtype=torch.float32,
        theta=theta,
        max_pos=[20],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=LTXRopeType.INTERLEAVED,
    )  # (1, N, 2048)
    a_cos = reshape_interleaved_to_bhnd(a_cos, num_heads=32)  # (1, 32, N, 64)
    a_sin = reshape_interleaved_to_bhnd(a_sin, num_heads=32)

    if audio_N > audio_N_real:
        head_dim = a_cos.shape[-1]
        a_cos_padded = torch.ones(1, 32, audio_N, head_dim)
        a_cos_padded[:, :, :audio_N_real, :] = a_cos
        a_sin_padded = torch.zeros(1, 32, audio_N, head_dim)
        a_sin_padded[:, :, :audio_N_real, :] = a_sin
        a_cos, a_sin = a_cos_padded, a_sin_padded

    sp_axis = parallel_config.sequence_parallel.mesh_axis
    tp_axis = parallel_config.tensor_parallel.mesh_axis
    return (
        bf16_tensor_2dshard(a_cos, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1}),
        bf16_tensor_2dshard(a_sin, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1}),
    )


def prepare_av_cross_pe(
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    audio_N: int,
    audio_N_real: int,
    *,
    theta: float,
    mesh_device: ttnn.MeshDevice,
    parallel_config: DiTParallelConfig,
    fps: float = 24.0,
    cross_pe_max_pos: int = 20,
    anchor_frames: list[int] | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Temporal-only cross positional embeddings for A↔V cross-attention.

    Reference: ``MultiModalTransformerArgsPreprocessor.prepare`` builds ``cross_pe`` from the
    temporal slice ``modality.positions[:, 0:1, :]`` at ``dim=audio_cross_attention_dim`` with
    ``max_pos=[cross_pe_max_pos]``. Video and audio share the scheme so a video token and an
    audio token at the same time get the same rotary phase (AV temporal sync).

    Returns 6 device tensors used by inner_step:
        (v_q_cos, v_q_sin)  — video Q in A→V cross-attn (SP×TP sharded). Also reused as the
                              video K rope in V→A (ring SDPA gathers the SP-sharded K).
        (a_q_cos, a_q_sin)  — audio Q in V→A cross-attn (SP×TP sharded).
        (a_k_cos, a_k_sin)  — audio K in A→V cross-attn (TP-only; K side after AllGather).
    """
    v_shape = VideoLatentShape(batch=1, channels=128, frames=latent_frames, height=latent_height, width=latent_width)
    v_coords = video_get_patch_grid_bounds(v_shape)
    v_positions = get_pixel_coords(v_coords, scale_factors=(8, 32, 32), causal_fix=True).float()
    v_positions[:, 0, ...] = v_positions[:, 0, ...] / fps  # temporal axis → seconds
    v_temporal = v_positions[:, 0:1, :]  # (1, 1, video_N, 2)
    if anchor_frames:
        # Append-token anchors ride the tail carrying their target frame's temporal phase, so the
        # video cross-PE stays length-matched to the extended video token sequence.
        if any(f < 0 or f >= latent_frames for f in anchor_frames):
            raise ValueError(f"anchor_frames {anchor_frames} out of range [0, {latent_frames})")
        hw = latent_height * latent_width
        v_temporal = torch.cat(
            [v_temporal] + [v_temporal[:, :, f * hw : (f + 1) * hw, :] for f in anchor_frames], dim=2
        )

    a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N_real, mel_bins=16)
    a_positions = audio_get_patch_grid_bounds(a_shape).float()  # (1, 1, audio_N_real, 2)

    rope_kwargs = dict(
        dim=2048,  # audio_cross_attention_dim — both sides share this
        out_dtype=torch.float32,
        theta=theta,
        max_pos=[cross_pe_max_pos],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=LTXRopeType.INTERLEAVED,
    )

    v_cos, v_sin = precompute_freqs_cis(v_temporal, **rope_kwargs)  # (1, video_N, 2048)
    a_cos, a_sin = precompute_freqs_cis(a_positions, **rope_kwargs)  # (1, audio_N_real, 2048)
    v_cos = reshape_interleaved_to_bhnd(v_cos, num_heads=32)  # (1, 32, video_N, 64)
    v_sin = reshape_interleaved_to_bhnd(v_sin, num_heads=32)
    a_cos = reshape_interleaved_to_bhnd(a_cos, num_heads=32)
    a_sin = reshape_interleaved_to_bhnd(a_sin, num_heads=32)

    v_cos, v_sin = pad_video_rope_sp(v_cos, v_sin, parallel_config.sequence_parallel.factor)

    if audio_N > audio_N_real:
        head_dim = a_cos.shape[-1]
        a_cos_padded = torch.ones(1, 32, audio_N, head_dim)
        a_cos_padded[:, :, :audio_N_real, :] = a_cos
        a_sin_padded = torch.zeros(1, 32, audio_N, head_dim)
        a_sin_padded[:, :, :audio_N_real, :] = a_sin
        a_cos, a_sin = a_cos_padded, a_sin_padded

    sp_axis = parallel_config.sequence_parallel.mesh_axis
    tp_axis = parallel_config.tensor_parallel.mesh_axis

    # Q-side: SP×TP sharded (matches the Q tensor layout post-attention QKV split).
    v_q_cos = bf16_tensor_2dshard(v_cos, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    v_q_sin = bf16_tensor_2dshard(v_sin, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    a_q_cos = bf16_tensor_2dshard(a_cos, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    a_q_sin = bf16_tensor_2dshard(a_sin, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

    # K-side: TP-only on heads (sequence is replicated after AllGather on K). Only the audio K
    # rope is needed (A→V gathers audio K); video K in V→A reuses the SP-sharded v_q rope.
    a_k_cos = bf16_tensor(a_cos, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
    a_k_sin = bf16_tensor(a_sin, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)

    return (v_q_cos, v_q_sin, a_q_cos, a_q_sin, a_k_cos, a_k_sin)
