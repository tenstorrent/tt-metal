# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""3-axis RoPE for the MiniMax-H3 visual VAE's ViT decoder.

Two things make this unlike every other RoPE in ``tt_dit``:

* ``rope_theta`` is **100.0**, not 10000.
* Only ``rope_dim_ratio * attention_head_dim = 48`` of each 64-wide head is rotated;
  the last 16 lanes pass through untouched.

The reference builds 24 angles (3 axes x 8 frequencies), ``.tile(2)`` duplicates them to
48, and a half-split ``chunk(2)`` pairs lane *i* with lane *i + 24*. Permuting the q/k
weight rows **once at load time** so those partners become adjacent --
``[0, 24, 1, 25, ..., 23, 47] + [48..63]`` -- turns that into an adjacent ``(2j, 2j+1)``
pairing. Build ``cos``/``sin`` in the same permuted basis (each angle duplicated onto its
lane pair, **cos=1 / sin=0 on the pass-through lanes 48..63**) and the rotation is exactly
``x*cos + rot90(x)*sin`` over the full head with no slice -- 48 is not tile-aligned
(48 % 32 = 16), so avoiding the slice is what makes this cheap.

Those permuted tables are already in the stacked ``(2j, 2j+1)`` basis that
``ttnn.experimental.rotary_embedding_llama`` consumes with the standard 32x32
``get_rot_transformation_mat`` -- that matrix applies the same ``(2j, 2j+1)`` per-tile
rotation as ``ttnn.alt_complex_rotate90``, so the decoder feeds the tables straight to the
fused op (one op per q/k). An earlier note here claimed the llama op ``cannot be used``;
that was wrong -- its pairing follows ``trans_mat``, not a fixed *i*<->*i+32*.

Q and K take the same permute, so ``Q K^T`` is unchanged; V is never permuted, so the
attention output is already in the normal basis and needs no inverse permute. The
``q_norm``/``k_norm`` are RMS over all 64 lanes with no learnable scale, hence
permutation-invariant -- nothing to fix up there either.
"""

from __future__ import annotations

import math

import torch


def rope_dim(attention_head_dim: int, rope_dim_ratio: float) -> int:
    return int(attention_head_dim * rope_dim_ratio)


def head_lane_permutation(attention_head_dim: int = 64, rope_dim_ratio: float = 0.75) -> torch.Tensor:
    """``[0, half, 1, half+1, ...]`` over the rotary lanes, then the pass-through lanes.

    Applied to the q/k weight rows within each head at load time, this turns the
    reference's ``(i, i + half)`` half-split pairing into the adjacent ``(2j, 2j+1)``
    pairing that ``alt_complex_rotate90`` implements.
    """
    rotary = rope_dim(attention_head_dim, rope_dim_ratio)
    half = rotary // 2
    interleaved = [index for j in range(half) for index in (j, j + half)]
    return torch.tensor(interleaved + list(range(rotary, attention_head_dim)), dtype=torch.long)


def position_grid(num_frames: int, height: int, width: int) -> torch.Tensor:
    """``(T*H*W, 3)`` coordinates, each axis length-normalised to ``[-1, 1)``.

    Matches the reference exactly: ``2 * (arange(0.5, size) / size) - 1`` per axis, then
    ``meshgrid(indexing="ij")`` so T is the slowest-varying axis.
    """
    grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32) / size) - 1.0 for size in (num_frames, height, width)]
    return torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)


def inverse_frequencies(rotary_dim: int, theta: float = 100.0, num_axes: int = 3) -> torch.Tensor:
    """``1 / theta ** arange(0, 1, 2 * num_axes / rotary_dim)`` -- 8 values at 48/3."""
    if rotary_dim % (2 * num_axes) != 0:
        raise ValueError(f"rotary_dim {rotary_dim} must be divisible by 2 * num_axes {2 * num_axes}")
    step = 2 * num_axes / rotary_dim
    return 1.0 / theta ** torch.arange(0, 1, step, dtype=torch.float32)


def rope_tables(
    num_frames: int,
    height: int,
    width: int,
    *,
    num_suffix_tokens: int,
    attention_head_dim: int = 64,
    rope_dim_ratio: float = 0.75,
    theta: float = 100.0,
    permuted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` of shape ``(num_patches + num_suffix_tokens, attention_head_dim)``.

    The suffix rows -- the learned register tokens plus the zero cls token -- get
    position id 0 on every axis, exactly as the reference does, which makes their
    ``cos = 1`` and ``sin = 0``.

    With ``permuted=True`` the tables are emitted in the interleaved basis that pairs
    with :func:`head_lane_permutation` and ``alt_complex_rotate90``, and the
    pass-through lanes carry ``cos = 1, sin = 0`` so they survive the rotation
    untouched. With ``permuted=False`` they are the reference's own layout, which is
    what the host-side equivalence check compares against.
    """
    rotary = rope_dim(attention_head_dim, rope_dim_ratio)
    inv_freq = inverse_frequencies(rotary, theta)

    positions = position_grid(num_frames, height, width)
    suffix = positions.new_zeros((num_suffix_tokens, 3))
    positions = torch.cat([positions, suffix], dim=0)

    # (N, 3, F) -> (N, 3F) -> tile to (N, 2*3F) == (N, rotary)
    angles = 2.0 * math.pi * positions[:, :, None] * inv_freq[None, None, :]
    angles = angles.flatten(1, 2).tile(2)

    if permuted:
        # Duplicate each angle adjacently so lanes (2j, 2j+1) share one angle, then pad
        # the pass-through lanes with 0 -- cos 0 = 1, sin 0 = 0.
        half = rotary // 2
        angles = angles[:, :half].repeat_interleave(2, dim=1)
        pad = attention_head_dim - rotary
        if pad:
            angles = torch.cat([angles, angles.new_zeros((angles.shape[0], pad))], dim=1)

    return angles.cos(), angles.sin()


def reference_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The reference's rotation, for use as a host-side oracle.

    ``x`` is ``(..., attention_head_dim)``; ``cos``/``sin`` are ``(..., rotary_dim)``.
    """
    rotary_dim = cos.shape[-1]
    x_rotary, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = x_rotary.chunk(2, dim=-1)
    rotated = torch.cat([-second, first], dim=-1)
    return torch.cat([x_rotary * cos + rotated * sin, x_pass], dim=-1)


def permuted_rotate(x_permuted: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The device form, in torch: ``x * cos + rot90(x) * sin`` over the full head.

    ``rot90`` here is ``ttnn.alt_complex_rotate90``'s golden: pairs ``(2j, 2j+1)`` map to
    ``(-x[2j+1], x[2j])``. No slicing, because the pass-through lanes carry cos=1, sin=0.
    """
    pairs = x_permuted.reshape(*x_permuted.shape[:-1], -1, 2)
    real, imag = pairs[..., 0:1], pairs[..., 1:2]
    rotated = torch.cat([-imag, real], dim=-1).flatten(-2)
    return x_permuted * cos + rotated * sin
