# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side RoPE math shared by the DiffVAE's deterministic stages and stage 5.

The two modules rotate in different device layouts: the deterministic stages in halves order, in
row-major volume form; stage 5 on upstream's interleaved pairs in TILE, its table factored into a
frame piece and a time piece, or fused in bricked site order. The angles under both are one thing:
per axis, position times a geometric series of rates, written into that axis's lanes. This module
is that one thing, plus the two lane-order matrices the layouts are built on. Torch only.
"""

from __future__ import annotations

import torch

ROPE_BASE = 10000.0


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    """Split ``head_dim`` across the (T, H, W) RoPE chunks (upstream's default; 64 -> (16, 24, 24))."""
    if head_dim % 8 != 0:
        msg = f"head_dim={head_dim} must be a multiple of 8 for the default split"
        raise ValueError(msg)
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return (d_t, d_hw, d_hw)


def inv_freqs(width: int, base: float = ROPE_BASE) -> torch.Tensor:
    """Per-pair rotation rates ``1 / base ** (2j / width)`` for the ``width // 2`` pairs of one axis chunk.

    The power is taken in float64 and cast after. Upstream computes these in numpy float64, and a
    float32 power lands the high-frequency rates across a bfloat16 rounding boundary, which shows
    as PCC drift once a table is cast to bf16.
    """
    exponents = torch.arange(0, width, 2, dtype=torch.float64) / width
    return (1.0 / base**exponents).to(torch.float32)


def axis_angles(positions: torch.Tensor, width: int, base: float = ROPE_BASE) -> torch.Tensor:
    """``positions`` ``(n,)`` -> angles ``(n, width // 2)``, float32: position times each pair's rate."""
    return positions.reshape(-1, 1).to(torch.float32) * inv_freqs(width, base).reshape(1, -1)


def interleaved_lanes(
    fn,
    axis: int,
    positions: torch.Tensor,
    dim_split: tuple[int, int, int],
    base: float = ROPE_BASE,
    *,
    ghost: torch.Tensor | None = None,
) -> torch.Tensor:
    """``fn`` (cos or sin) of one axis's angles in that axis's lanes, zero elsewhere: ``(n, head_dim)``.

    Upstream's lane order: pairs are ``repeat_interleave``d so pair ``j`` lands on lanes
    ``(2j, 2j + 1)`` of the axis chunk. Summing the three axes' results gives the full row. Rows
    flagged in ``ghost`` are zeroed: a ghost site has no position, and a zero row makes its rotation
    zero rather than a rotation by a made-up angle.
    """
    width = dim_split[axis]
    offset = sum(dim_split[:axis])
    angles = axis_angles(positions.reshape(-1), width, base)
    rows = torch.zeros(angles.shape[0], sum(dim_split), dtype=torch.float32)
    rows[:, offset : offset + width] = fn(angles).repeat_interleave(2, dim=-1)
    if ghost is not None:
        rows[ghost.reshape(-1)] = 0
    return rows


def rope_permutation(dim_split: tuple[int, int, int]) -> torch.Tensor:
    """``head_dim`` reordering that turns upstream's interleaved pairs into two halves.

    Upstream rotates adjacent dim pairs ``(d0,d1), (d2,d3), ...`` within each axis chunk, which on
    device would need a stride-2 gather per rotation. Attention only sees ``q·k``, so permuting
    ``head_dim`` identically in q and k is invisible in the output -- and reordering to
    ``[all first-of-pair, all second-of-pair]`` makes RoPE the contiguous
    ``(x1*cos - x2*sin, x1*sin + x2*cos)``. Verified bit-identical to upstream.

    The deterministic stages fold this into the q/k projection rows and the q_norm/k_norm weights
    at load time, so it is free at runtime. RMSNorm tolerates it because its scale is over all dims,
    hence permutation-invariant, provided its learned weight is permuted the same way.
    """
    evens, odds, offset = [], [], 0
    for width in dim_split:
        evens.extend(range(offset, offset + width, 2))
        odds.extend(range(offset + 1, offset + width, 2))
        offset += width
    return torch.tensor(evens + odds)


def pair_swap_matrix(head_dim: int) -> torch.Tensor:
    """``x @ P`` maps adjacent pairs ``(x0, x1)`` to ``(-x1, x0)``.

    Stage 5's rotation on the interleaved layout: a matmul rather than slice-and-concat because the
    per-axis RoPE chunks start at lanes 0/16/40 for head_dim 64 -- none of the odd/even sub-slices
    land on a tile boundary, so every alternative needs a row-major detour per chunk.
    """
    p = torch.zeros(head_dim, head_dim, dtype=torch.float32)
    for j in range(head_dim // 2):
        p[2 * j + 1, 2 * j] = -1.0
        p[2 * j, 2 * j + 1] = 1.0
    return p
