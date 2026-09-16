# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device-tensor scaffolding shared by the DiffVAE's deterministic stages and stage 5.

Everything here is about ttnn bookkeeping, not the model: freeing operands once an op has read
them, moving a tensor through ROW_MAJOR to reshape across the tile grid, slicing without copying
when the slice is the whole tensor, and the mesh geometry both halves keep asking for.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import ttnn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Tile height and width. A TILE-layout tensor pads both of its last two dims to this.
TILE = ttnn.TILE_SIZE


def mesh_axis_size(mesh_device: ttnn.MeshDevice, axis: int) -> int:
    """How many chips the mesh has along ``axis``."""
    return int(list(mesh_device.shape)[axis])


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


def consume(x: ttnn.Tensor, op: Callable, *args, **kwargs) -> ttnn.Tensor:
    """``op(x, *args, **kwargs)``, freeing ``x`` unless ``op`` handed it straight back.

    Only for ops that copy: a view over ``x``'s buffer would be freed under the result.
    """
    out = op(x, *args, **kwargs)
    if out is not x:
        ttnn.deallocate(x)
    return out


def consume_all(op: Callable, *tensors: ttnn.Tensor) -> ttnn.Tensor:
    """``op(*tensors)``, freeing every operand. For binary ops whose inputs are both temporaries."""
    out = op(*tensors)
    for tensor in tensors:
        ttnn.deallocate(tensor)
    return out


def release_intermediates(tensors: Sequence[ttnn.Tensor], *, keep: ttnn.Tensor) -> None:
    """Free every distinct buffer among ``tensors`` except the one ``keep`` is using.

    ``ttnn.reshape`` may hand back a new Python object that is a VIEW over its input's buffer, and
    ``ttnn.deallocate`` defaults to ``force=True``, so an ``a is not b`` guard can free memory a
    live tensor still reads. Compare buffers, which is what is being freed.
    """
    seen = {keep.buffer_address()} if keep.is_allocated() else set()
    for tensor in tensors:
        if not tensor.is_allocated():
            continue
        address = tensor.buffer_address()
        if address in seen:
            continue
        seen.add(address)
        ttnn.deallocate(tensor)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def retile(x: ttnn.Tensor, shape: Sequence[int]) -> ttnn.Tensor:
    """Reshape across the tile grid via ROW_MAJOR and return TILE; ``ttnn.reshape`` will not re-tile
    the last two dims. Returns ``x`` itself when the shape already matches; never frees ``x``."""
    shape = tuple(shape)
    if tuple(x.shape) == shape:
        return x
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.to_layout(ttnn.reshape(rm, shape), ttnn.TILE_LAYOUT)
    # A ROW_MAJOR input is returned as-is by to_layout, and that one belongs to the caller.
    if rm is not x:
        ttnn.deallocate(rm)
    return out


def to_row_major(x: ttnn.Tensor, shape: Sequence[int]) -> ttnn.Tensor:
    """Reshape and leave the result in ROW_MAJOR.

    TILE pads both of the last two dims to 32, so a trailing ``(num_heads, head_dim)`` in TILE
    costs many times its own size; the attention gathers rows in ROW_MAJOR anyway.
    """
    return ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), tuple(shape))


def slice_rows(x: ttnn.Tensor, lo: int, hi: int) -> ttnn.Tensor:
    """Rows ``[lo, hi)`` along the second-to-last dim, or ``x`` itself if that is every row."""
    shape = tuple(x.shape)
    if (lo, hi) == (0, shape[-2]):
        return x
    starts = [0] * len(shape)
    stops = list(shape)
    starts[-2], stops[-2] = lo, hi
    return ttnn.slice(x, starts, stops)


def slice_last(x: ttnn.Tensor, start: int, stop: int) -> ttnn.Tensor:
    """Slice the last dim. Callers keep ``start``/``stop`` tile-aligned."""
    starts = [0] * (len(x.shape) - 1) + [start]
    stops = [*list(x.shape)[:-1], stop]
    return ttnn.slice(x, starts, stops)


def align_down(value: int, step: int) -> int:
    """Largest multiple of ``step`` that is <= ``value``. The counterpart of ``utils.ltx.ceil_to``."""
    return value if step <= 1 else value - (value % step)


def pad_dim(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Zero-pad a host weight along ``dim`` up to ``size``. Host side, for tile-aligning features."""
    extra = size - tensor.shape[dim]
    if extra == 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = extra
    return torch.cat([tensor, tensor.new_zeros(shape)], dim=dim)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


def wshard(x: ttnn.Tensor, volume: tuple[int, int, int], *, sp_axis: int) -> ttnn.Tensor:
    """Reshard a replicated TILE tensor whose row axis is ``T*H*W`` into this chip's W-band.

    Leading dims are kept; the rows become ``T*H*(W/sp)``. **Consumes** ``x``.
    """
    t, h, w = volume
    lead = tuple(x.shape)[:-2]
    channels = int(x.shape[-1])
    sp = mesh_axis_size(x.device(), sp_axis)
    rm = consume(x, ttnn.to_layout, ttnn.ROW_MAJOR_LAYOUT)
    # Rank 4 with the leading dims folded into T: W is dim 2 either way, and that is all the
    # partition needs to know.
    vol = ttnn.reshape(rm, (math.prod(lead) * t, h, w, channels))
    band = ttnn.mesh_partition(vol, dim=2, cluster_axis=sp_axis)
    ttnn.deallocate(rm)
    flat = ttnn.reshape(band, (*lead, t * h * (w // sp), channels))
    return ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
