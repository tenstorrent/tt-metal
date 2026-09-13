# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Natural <-> Bricked token order for 3D neighborhood attention.

A hardware tile is 32 sites wide, so a tile is the smallest thing that can be read. In
natural (time, height, width) row-major order those 32 sites are a ``1 x 1 x 32`` pencil
along width, and a query at one end of the pencil shares almost none of its context window
with a query at the other. Bricking reorders the tokens so 32 consecutive sites are a
compact 3D box instead -- ``2 x 4 x 4`` for a cubic context window -- which is the same read
cost spent on sites that are actually useful together.

Two consequences, both measured rather than assumed (see the module tests):

* the union of the context windows across one tile shrinks (``11x11x11`` at stride 1:
  ``11x11x42`` = 5082 sites in natural order, ``12x14x14`` = 2352 bricked), and
* the window becomes a few long memory runs instead of many short ones.

Applied ONCE at stage entry and once at exit, not per block: every other operation in a
DiffVAE block (norms, modulation, SwiGLU, residuals) is per-token and therefore
permutation-equivariant, so all eight blocks run in bricked order. RoPE tables must be
permuted the same way, once, at init.

This module is the sole definition of the ordering. The SDPA kernels do not convert -- they
assume it, addressing one page per brick in ``neighborhood_chunk_layout.hpp``. So the formula
below and that addressing must agree, with nothing in the type system tying them together::

    brick_index         = (time // brick_time) * (bricks_height * bricks_width)
                        + (height // brick_height) * bricks_width
                        + (width // brick_width)
    site_index_in_brick = (time % brick_time) * (brick_height * brick_width)
                        + (height % brick_height) * brick_width
                        + (width % brick_width)
    bricked_index       = brick_index * SITES_PER_BRICK + site_index_in_brick

The reshape-and-permute below is that formula, realised by moving axes: flattening
``(bricks..., sites_in_brick...)`` in that order produces exactly this index. If one changes,
the other must. There is a test that pins them together.

Composed ttnn ops rather than a dedicated kernel, because measurement says they are free:
0.48 ms for a device's 52 MB share of stage 5 at 1080p, against a decode measured in
seconds. Use ROW_MAJOR -- TILE layout is ~15x slower here, because the reshape puts small
extents (2, 4, 4) on the tiled axes and each pads out to 32.
"""

from __future__ import annotations

import math

import ttnn

#: Sites in one brick, i.e. rows in one hardware tile.
SITES_PER_BRICK = 32

#: [batch, bricks_t, brick_t, bricks_h, brick_h, bricks_w, brick_w, channels]
#:   -> [batch, bricks_t, bricks_h, bricks_w, brick_t, brick_h, brick_w, channels]
_TO_BRICKED_ORDER = (0, 1, 3, 5, 2, 4, 6, 7)
_TO_NATURAL_ORDER = (0, 1, 4, 2, 5, 3, 6, 7)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def padded_volume(volume: tuple[int, int, int], brick: tuple[int, int, int]) -> tuple[int, int, int]:
    """``volume`` rounded up so every axis holds whole bricks.

    A volume that already divides evenly is returned unchanged, and no padding op runs.
    Stage 5 at 1080p is ``(25, 272, 480)``: with brick ``(2, 4, 4)`` the time axis needs one
    ghost frame, while brick ``(1, 4, 8)`` divides exactly and needs none.
    """
    return tuple(
        _ceil_div(volume_extent, brick_extent) * brick_extent for volume_extent, brick_extent in zip(volume, brick)
    )


def brick_grid(volume: tuple[int, int, int], brick: tuple[int, int, int]) -> tuple[int, int, int]:
    """``(T_br, H_br, W_br)`` -- how many bricks along each axis, after ghost padding."""
    padded_time, padded_height, padded_width = padded_volume(volume, brick)
    return padded_time // brick[0], padded_height // brick[1], padded_width // brick[2]


def brick_count(volume: tuple[int, int, int], brick: tuple[int, int, int]) -> int:
    """How many bricks -- and therefore how many tiles of sites -- the volume occupies."""
    return math.prod(brick_grid(volume, brick))


def sites_per_t_brick(volume: tuple[int, int, int], brick: tuple[int, int, int]) -> int:
    """Sites in one T-brick slab: ``H_br * W_br * 32``. A T-range is a contiguous slice
    of a bricked tensor iff its bounds are multiples of ``brick[0]``."""
    _, bricks_height, bricks_width = brick_grid(volume, brick)
    return bricks_height * bricks_width * SITES_PER_BRICK


def _require_row_major(tensor: ttnn.Tensor, function_name: str) -> None:
    if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(
            f"{function_name} needs ROW_MAJOR: the reshape puts brick extents on the tiled axes, "
            f"where TILE layout pads each of them to 32 and runs ~15x slower. Got {tensor.layout}."
        )


def to_bricked_grid(tensor: ttnn.Tensor, *, volume: tuple[int, int, int], brick: tuple[int, int, int]) -> ttnn.Tensor:
    """``[batch, time, height, width, channels]`` -> ``[batch, T_br, H_br, W_br, 32*channels]``.

    Brick index is outermost, row-major -- the same order as the flattened form, but with
    ``T_br`` and ``W_br`` still as axes. That is what lets a W halo exchange be ``neighbor_pad``
    on dim 3 (whole bricks, no permute) and a T-band slice be a contiguous cut on dim 1,
    provided the band bounds are multiples of ``brick[0]``.
    """
    _require_row_major(tensor, "to_bricked_grid")

    batch_count = tensor.shape[0]
    channel_count = tensor.shape[-1]
    brick_time, brick_height, brick_width = brick
    padded_time, padded_height, padded_width = padded_volume(volume, brick)
    bricks_t, bricks_h, bricks_w = brick_grid(volume, brick)

    # Ghost sites are appended by concatenating zeros rather than with ttnn.pad: pad only
    # reaches the lowest 3 dimensions of a rank>4 tensor, and the time axis (dim 1 here) is
    # exactly the one stage 5 needs -- 25 frames against a brick 2 deep.
    padded = (padded_time, padded_height, padded_width)
    for axis, (current, target) in enumerate(zip(volume, padded), start=1):
        if target == current:
            continue
        ghost_shape = list(tensor.shape)
        ghost_shape[axis] = target - current
        ghost = ttnn.zeros(ghost_shape, dtype=tensor.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=tensor.device())
        tensor = ttnn.concat([tensor, ghost], dim=axis)
        ttnn.deallocate(ghost)

    split_into_bricks = ttnn.reshape(
        tensor,
        (
            batch_count,
            bricks_t,
            brick_time,
            bricks_h,
            brick_height,
            bricks_w,
            brick_width,
            channel_count,
        ),
    )
    bricks_outermost = ttnn.permute(split_into_bricks, _TO_BRICKED_ORDER)
    return ttnn.reshape(
        bricks_outermost,
        (batch_count, bricks_t, bricks_h, bricks_w, SITES_PER_BRICK * channel_count),
    )


def to_bricked(tensor: ttnn.Tensor, *, volume: tuple[int, int, int], brick: tuple[int, int, int]) -> ttnn.Tensor:
    """``[batch, time, height, width, channels]`` -> ``[batch, sites_bricked, channels]``.

    ``sites_bricked`` is ``brick_count(volume, brick) * SITES_PER_BRICK``, which exceeds
    ``time * height * width`` when the volume does not divide into whole bricks. Those ghost
    sites are zero. They are never inside any context window -- window placement uses the
    true ``volume`` -- but they do sit inside edge bricks, so a whole-brick read will pull
    them and the kernel must mask them out.
    """
    grid = to_bricked_grid(tensor, volume=volume, brick=brick)
    batch_count = grid.shape[0]
    channel_count = grid.shape[-1] // SITES_PER_BRICK
    return ttnn.reshape(grid, (batch_count, brick_count(volume, brick) * SITES_PER_BRICK, channel_count))


def from_bricked_grid(tensor: ttnn.Tensor, *, volume: tuple[int, int, int], brick: tuple[int, int, int]) -> ttnn.Tensor:
    """``[batch, T_br, H_br, W_br, 32*channels]`` -> ``[batch, time, height, width, channels]``.

    Exact inverse of :func:`to_bricked_grid`, ghost sites cropped back off.
    """
    _require_row_major(tensor, "from_bricked_grid")

    batch_count = tensor.shape[0]
    channel_count = tensor.shape[-1] // SITES_PER_BRICK
    brick_time, brick_height, brick_width = brick
    padded_time, padded_height, padded_width = padded_volume(volume, brick)
    bricks_t, bricks_h, bricks_w = brick_grid(volume, brick)

    split_into_bricks = ttnn.reshape(
        tensor,
        (
            batch_count,
            bricks_t,
            bricks_h,
            bricks_w,
            brick_time,
            brick_height,
            brick_width,
            channel_count,
        ),
    )
    axes_interleaved = ttnn.permute(split_into_bricks, _TO_NATURAL_ORDER)
    natural = ttnn.reshape(axes_interleaved, (batch_count, padded_time, padded_height, padded_width, channel_count))

    if (padded_time, padded_height, padded_width) != tuple(volume):
        natural = natural[:, : volume[0], : volume[1], : volume[2], :]
    return natural


def to_natural(tensor: ttnn.Tensor, *, volume: tuple[int, int, int], brick: tuple[int, int, int]) -> ttnn.Tensor:
    """``[batch, sites_bricked, channels]`` -> ``[batch, time, height, width, channels]``.

    Exact inverse of :func:`to_bricked`, ghost sites cropped back off.
    """
    _require_row_major(tensor, "to_natural")

    batch_count = tensor.shape[0]
    channel_count = tensor.shape[-1]
    bricks_t, bricks_h, bricks_w = brick_grid(volume, brick)
    grid = ttnn.reshape(
        tensor,
        (batch_count, bricks_t, bricks_h, bricks_w, SITES_PER_BRICK * channel_count),
    )
    return from_bricked_grid(grid, volume=volume, brick=brick)
