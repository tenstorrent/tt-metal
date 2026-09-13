# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The two plans behind 3D neighborhood attention in the LTX-2.5 DiffVAE decoder.

Every block attends over a local 3D window rather than the whole volume (``(3,7,7)`` and
``(3,5,5)`` in the deterministic stages, ``(11,11,11)`` in the diffusion stage). The window rule
is NATTEN's: the window keeps its SIZE and slides inward at a boundary, so a query at index 0
attends to ``[0, K)``, never to a truncated ``[0, K//2]``. That rule is defined once, in
``neighborhood_reference.context_window_origin``; everything here consumes it.

Two executors run that attention, and each has a plan -- the geometry that depends on the volume,
window, stride and mesh but on no weights, so it is built once per shape and reused by every block:

* **The linear-order plan** (``plan_na3d`` -> ``NA3DPlan`` -> ``build_device_plan`` ->
  ``NA3DDevicePlan``). Tokens stay in natural row-major order. Query tiles are grouped by window
  geometry (at stride 1 an axis has three regimes, so a volume collapses to at most 27 groups), each
  group gets one additive mask, and the device plan uploads per-group gather indices, optionally
  split across the mesh (``NA3DShard``: tiles over one axis, query rows over the other, K/V
  replicated). ``na3d_torch`` executes such a plan on the host and is the tiled torch oracle the
  tests use where the dense reference cannot fit. ``NA3DPlan.describe()`` prints a plan as prose.

* **The bricked plan** (``cached_bricked_plan``). Tokens are permuted into 32-site bricks
  (``neighborhood_permute``) so a window is a small box of whole tiles. The chooser
  (``_choose_sharded_brick``) picks the brick, the query chunk is the largest brick run that still
  shares one window (``_query_chunk_bricks``), and the C++ planner (``ttnn.transformer.
  neighborhood_plan``) returns the gather table one shard sees. This module wraps that per shard,
  checks the shards agree on every shape (one program serves them all), uploads the stacked gather
  origins, and builds the mask tables the reader keeps resident: the RELATIVE table at stride 1
  (``_build_relative_masks``), the per-REGIME sets under a GNA stride (``_build_regime_masks``).

The executors live in ``neighborhood_attention.py`` and import from here; nothing here imports
them.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

import ttnn

from ..utils.tensor import from_torch
from .neighborhood_permute import SITES_PER_BRICK
from .neighborhood_reference import context_window_origin

# Cap on one tile group's [Nq, Nk] score block. Bounds both the additive mask allocation and
# the score materialization; the tile search shrinks axes until the product fits.
DEFAULT_SCORE_BUDGET = 2**22


def window_bounds(length: int, kernel: int, stride: int = 1) -> tuple[list[int], list[int]]:
    """Per-index ``(start, end)`` of the attended window along one axis.

    NATTEN's constant-size inward-shifted window: the start is the query index less half the
    kernel, clamped so the window never leaves ``[0, length)``. When the axis is shorter than the
    kernel every query attends to the whole axis.

    ``stride`` is GNA's query-group size: runs of ``stride`` queries share one window, placed by
    ``context_window_origin`` -- the rule the bricked op ships (``window_origin_on_axis`` in
    ``neighborhood_window_rule.hpp``), which is NATTEN's default: the group's centre-most site leads,
    taken from the right for an even-sized group. No brick snapping: pass ``brick=`` to the dense
    reference for that. ``stride=1`` is standard neighborhood attention, every query centred on its
    own window.
    """
    kernel = min(kernel, length)
    starts = [context_window_origin(i // stride, stride, kernel, length) for i in range(length)]
    return starts, [s + kernel for s in starts]


def _tile_lengths(dims: tuple[int, int, int], kernels: tuple[int, int, int], budget: int) -> tuple[int, int, int]:
    """Per-axis query-tile lengths keeping one tile's score block under ``budget``.

    Halves whichever axis is largest relative to its kernel, since that is the axis whose
    span is cheapest to shrink: a tile of length ``t`` spans ``t + k - 1`` keys, so the waste
    factor is ``(t + k - 1) / k`` and shrinking a long axis with a small kernel helps most.
    """
    tiles = list(dims)

    def score_block(candidate: list[int]) -> int:
        n_q = math.prod(candidate)
        n_k = math.prod(min(d, t + k - 1) for t, k, d in zip(candidate, kernels, dims))
        return n_q * n_k

    while score_block(tiles) > budget and max(tiles) > 1:
        axis = max(range(3), key=lambda a: tiles[a] / kernels[a])
        if tiles[axis] <= 1:
            break
        tiles[axis] = max(1, (tiles[axis] + 1) // 2)
    return tiles[0], tiles[1], tiles[2]


AxisGeometry = tuple[tuple[int, ...], tuple[int, ...]]
"""Per-axis window bounds of one query tile, relative to that tile's key span."""


@dataclass(frozen=True)
class TileGroup:
    """Query tiles that share one window geometry, and therefore one additive mask.

    Grouping matters: at stride 1 the inward-shift rule gives an axis only three regimes (leading
    clamp, interior slide, trailing clamp), so a full volume collapses to at most 27 distinct masks
    in 3D no matter how many tiles there are. Under a GNA stride the interior regime instead repeats
    with period ``stride``, so the count depends on how the tiling lines up with the groups -- still
    bounded and still correct, just no longer 27.
    """

    geometry: tuple[AxisGeometry, AxisGeometry, AxisGeometry]
    query_slices: tuple[tuple[slice, slice, slice], ...]
    key_slices: tuple[tuple[slice, slice, slice], ...]
    n_queries: int
    n_keys: int


@dataclass(frozen=True)
class NA3DPlan:
    dims: tuple[int, int, int]
    kernels: tuple[int, int, int]
    tile: tuple[int, int, int]
    groups: tuple[TileGroup, ...]
    stride: tuple[int, int, int] = (1, 1, 1)

    def describe(self, max_groups: int | None = None) -> str:
        """The plan as prose: what is tiled how, how many distinct masks that makes, and what it costs.

        One line per group, largest first, each naming the per-axis regime in words -- ``slides``
        (the interior: each query tile's window starts one site later than the last), ``low edge``
        / ``high edge`` (every query in the tile shares the clamped window), ``whole axis`` (the axis
        is no wider than the window). ``max_groups`` truncates the listing; the totals never are.
        """
        axis_names = ("t", "h", "w")
        dims, kernels, tile = self.dims, self.kernels, self.tile
        tiles_per_axis = [-(-extent // step) for extent, step in zip(dims, tile)]
        total_tiles = math.prod(tiles_per_axis)
        sites = math.prod(dims)

        lines = [
            f"NA3D plan over volume (t,h,w)={dims}: {sites:,} sites, window {kernels}"
            + (
                f" ({', '.join(n for n, k, d in zip(axis_names, kernels, dims) if k == d)} attended in full)"
                if any(k == d for k, d in zip(kernels, dims))
                else ""
            )
            + (f", GNA stride {self.stride}" if self.stride != (1, 1, 1) else ""),
            f"  query tile {tile}: {' x '.join(map(str, tiles_per_axis))} = {total_tiles} tile{'s' * (total_tiles != 1)}, "
            f"{len(self.groups)} distinct window geometr{'ies' if len(self.groups) != 1 else 'y'} (one additive mask each)",
        ]

        # Cost: the score block every tile actually computes, against the dense volume and the
        # ideal (every query sees exactly its window).
        planned_scores = sum(len(group.query_slices) * group.n_queries * group.n_keys for group in self.groups)
        ideal_scores = sites * math.prod(kernels)
        lines.append(
            f"  scores: {planned_scores:,} planned vs {ideal_scores:,} ideal (x{planned_scores / ideal_scores:.2f} "
            f"window waste) vs {sites * sites:,} dense ({sites * sites / planned_scores:,.0f}x sparser)"
        )

        def regime(group: TileGroup, axis: int) -> str:
            starts, _ = group.geometry[axis]
            key_slice = group.key_slices[0][axis]
            span = key_slice.stop - key_slice.start
            if span <= kernels[axis] and key_slice.start == 0 and key_slice.stop == dims[axis]:
                return "whole axis"
            if len(set(starts)) == 1:
                return "low edge" if key_slice.start == 0 else "high edge"
            step = self.stride[axis]
            return "slides" if step == 1 else f"slides in steps of {step}"

        ordered = sorted(self.groups, key=lambda g: (-len(g.query_slices), -g.n_keys))
        shown = ordered if max_groups is None else ordered[:max_groups]
        for index, group in enumerate(shown, 1):
            regimes = ", ".join(f"{name}={regime(group, axis)}" for axis, name in enumerate(axis_names))
            q_shape = " x ".join(str(s.stop - s.start) for s in group.query_slices[0])
            k_shape = " x ".join(str(s.stop - s.start) for s in group.key_slices[0])
            lines.append(
                f"  group {index:>2}: {len(group.query_slices):>4} tile{'s' if len(group.query_slices) != 1 else ''}, "
                f"{q_shape} = {group.n_queries:,} queries "
                f"each see {k_shape} = {group.n_keys:,} keys  [{regimes}]"
            )
        if len(shown) < len(ordered):
            rest = ordered[len(shown) :]
            lines.append(f"  ... {len(rest)} more groups covering {sum(len(g.query_slices) for g in rest)} tiles")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def plan_na3d(
    dims: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    *,
    budget: int = DEFAULT_SCORE_BUDGET,
    stride: tuple[int, int, int] = (1, 1, 1),
) -> NA3DPlan:
    """Group the volume's query tiles by window geometry."""
    kernels = tuple(min(k, d) for k, d in zip(kernel_size, dims))
    bounds = [window_bounds(d, k, s) for d, k, s in zip(dims, kernels, stride)]
    tile = _tile_lengths(dims, kernels, budget)

    # Per axis: for each tile, the key span it needs and its bounds relative to that span.
    per_axis: list[list[tuple[slice, slice, AxisGeometry]]] = []
    for axis in range(3):
        length, step = dims[axis], tile[axis]
        starts, ends = bounds[axis]
        entries = []
        for begin in range(0, length, step):
            stop = min(begin + step, length)
            span_start, span_stop = starts[begin], ends[stop - 1]
            geometry = (
                tuple(s - span_start for s in starts[begin:stop]),
                tuple(e - span_start for e in ends[begin:stop]),
            )
            entries.append((slice(begin, stop), slice(span_start, span_stop), geometry))
        per_axis.append(entries)

    grouped: dict[tuple[AxisGeometry, ...], list[tuple[tuple[slice, ...], tuple[slice, ...]]]] = {}
    for t_q, t_k, t_geometry in per_axis[0]:
        for h_q, h_k, h_geometry in per_axis[1]:
            for w_q, w_k, w_geometry in per_axis[2]:
                key = (t_geometry, h_geometry, w_geometry)
                grouped.setdefault(key, []).append(((t_q, h_q, w_q), (t_k, h_k, w_k)))

    groups = []
    for geometry, members in grouped.items():
        q_slices, k_slices = zip(*members)
        n_queries = math.prod(s.stop - s.start for s in q_slices[0])
        n_keys = math.prod(s.stop - s.start for s in k_slices[0])
        groups.append(
            TileGroup(
                geometry=geometry,
                query_slices=tuple(q_slices),
                key_slices=tuple(k_slices),
                n_queries=n_queries,
                n_keys=n_keys,
            )
        )
    return NA3DPlan(dims=dims, kernels=kernels, tile=tile, groups=tuple(groups), stride=stride)


def group_mask(group: TileGroup, *, dtype: torch.dtype, device: torch.device | str = "cpu") -> torch.Tensor:
    """Additive ``[1, 1, Nq, Nk]`` mask for one group: 0 where visible, -inf where not."""
    visible_per_axis = []
    for starts, ends in group.geometry:
        start = torch.tensor(starts, device=device)
        end = torch.tensor(ends, device=device)
        key_index = torch.arange(int(end.max()), device=device)
        visible_per_axis.append((key_index[None, :] >= start[:, None]) & (key_index[None, :] < end[:, None]))

    # Outer-product the three axes into [Tq,Hq,Wq, Tk,Hk,Wk], then flatten to [Nq, Nk].
    visible = (
        visible_per_axis[0][:, None, None, :, None, None]
        & visible_per_axis[1][None, :, None, None, :, None]
        & visible_per_axis[2][None, None, :, None, None, :]
    )
    mask = torch.zeros((group.n_queries, group.n_keys), dtype=dtype, device=device)
    mask.masked_fill_(~visible.reshape(group.n_queries, group.n_keys), torch.finfo(dtype).min)
    return mask.reshape(1, 1, group.n_queries, group.n_keys)


def na3d_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
    *,
    scale: float | None = None,
    plan: NA3DPlan | None = None,
    stride: tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    """Host executor for a plan. ``q``/``k``/``v`` are ``(B, T, H, W, NH, HD)``.

    Pass ``scale=1.0`` when the caller has already scaled Q, which is what the DiffVAE
    blocks do — applying it again here would square the factor.

    ``stride`` is ignored when an explicit ``plan`` is given; the plan already carries its own.
    """
    batch, t, h, w, heads, head_dim = q.shape
    if scale is None:
        scale = head_dim**-0.5
    if scale != 1.0:
        q = q * scale
    if plan is None:
        plan = plan_na3d((t, h, w), kernel_size, stride=stride)

    out = torch.empty_like(v)
    for group in plan.groups:
        mask = group_mask(group, dtype=q.dtype, device=q.device)
        for q_slice, k_slice in zip(group.query_slices, group.key_slices):
            # (B, tq, th, tw, NH, HD) -> (B, NH, Nq, HD): attention wants heads ahead of seq.
            q_tile = q[:, q_slice[0], q_slice[1], q_slice[2]]
            tile_shape = q_tile.shape[1:4]
            q_flat = q_tile.permute(0, 4, 1, 2, 3, 5).reshape(batch, heads, group.n_queries, head_dim)
            k_flat = (
                k[:, k_slice[0], k_slice[1], k_slice[2]]
                .permute(0, 4, 1, 2, 3, 5)
                .reshape(batch, heads, group.n_keys, head_dim)
            )
            v_flat = (
                v[:, k_slice[0], k_slice[1], k_slice[2]]
                .permute(0, 4, 1, 2, 3, 5)
                .reshape(batch, heads, group.n_keys, head_dim)
            )
            attended = torch.nn.functional.scaled_dot_product_attention(
                q_flat, k_flat, v_flat, attn_mask=mask, scale=1.0
            )
            out[:, q_slice[0], q_slice[1], q_slice[2]] = attended.view(batch, heads, *tile_shape, head_dim).permute(
                0, 2, 3, 4, 1, 5
            )
    return out


def _flat_indices(dims: tuple[int, int, int], block: tuple[slice, slice, slice]) -> torch.Tensor:
    """Row indices into a ``T*H*W``-row table for one (t, h, w) block, in row-major order."""
    t, h, w = dims
    grid = torch.arange(t * h * w).reshape(t, h, w)
    return grid[block[0], block[1], block[2]].reshape(-1)


@dataclass(frozen=True)
class NA3DShard:
    """How a plan splits its query work across a 2D mesh.

    Two independent splits, both on the query side: a group's tiles across one mesh axis, and
    the query rows *within* a tile across the other. Keys and values stay replicated, so every
    tile can reach the whole of its window without a neighbour's help and there is no halo
    exchange. The trade is that the q/k/v tables stay full size on every chip: what shrinks is
    the arithmetic and the per-call gather, not the resident volume.

    Sharding here is a property of the *data* alone. Every chip walks the same groups in the
    same order and issues the same ops; only the contents of the index tensors and the masks
    differ. That is what makes this safe on a mesh that dispatches one program to every chip —
    a split that gave one chip more tile groups than another would not be.
    """

    tile_axis: int
    tile_factor: int
    row_axis: int
    row_factor: int

    @classmethod
    def for_mesh(cls, mesh_device) -> NA3DShard | None:
        """The default split for ``mesh_device``, or ``None`` if it has nothing to split.

        Tiles take the longer mesh axis because that is the axis with something to divide: a
        group holds thousands of tiles against a few hundred query rows per tile, and the
        shipped stage-5 geometry gives 1624 tiles, an exact multiple of 8.
        """
        shape = list(mesh_device.shape)
        if len(shape) != 2 or math.prod(shape) == 1:
            return None
        tile_axis = 0 if shape[0] >= shape[1] else 1
        row_axis = 1 - tile_axis
        return cls(tile_axis=tile_axis, tile_factor=shape[tile_axis], row_axis=row_axis, row_factor=shape[row_axis])


def _pad_by_duplication(x: torch.Tensor, *, dim: int, multiple: int) -> torch.Tensor:
    """Extend ``dim`` to a multiple of ``multiple`` by repeating its last entry.

    Duplication rather than a sentinel value, because it makes the padding *correct* instead of
    merely tolerable: a duplicated tile recomputes a tile that already exists and a duplicated
    query row recomputes a row that already exists, so both emit the values they would have
    emitted anyway. Nothing downstream has to know which rows came from padding, and a mistake
    in the padding arithmetic costs redundant work rather than wrong pixels.
    """
    extent = x.shape[dim]
    remainder = extent % multiple
    if remainder == 0:
        return x
    repeats = [1] * x.dim()
    repeats[dim] = multiple - remainder
    return torch.cat([x, x.narrow(dim, extent - 1, 1).repeat(repeats)], dim=dim)


@dataclass(frozen=True)
class NA3DGroup:
    """One tile group's uploaded gather indices and mask, with the extents one chip sees.

    ``local_tiles`` and ``local_queries`` are per chip; a replicated plan has them equal to the
    group's own (padded) counts. The factors are per group rather than per plan because a group
    too small to split keeps a factor of 1 while its neighbours are split — still uniform across
    chips, which is all the mesh requires.
    """

    query_indices: ttnn.Tensor
    key_indices: ttnn.Tensor
    mask: ttnn.Tensor
    local_tiles: int
    local_queries: int
    n_keys: int
    tile_factor: int
    row_factor: int


def _emitted_order(
    padded_rows: list[tuple[torch.Tensor, int, int]],
    shard: NA3DShard | None,
) -> list[torch.Tensor]:
    """Volume row index per output row, in the order the mesh gathers produce them.

    Every group's local result is concatenated into one stack per chip and that stack is gathered
    once per mesh axis, rather than gathering each group separately: a group-at-a-time gather
    needs two CCL programs per group, and at ~50 groups per geometry compiling them costs more
    than the attention itself. The price is that the output is no longer in group order — an
    all-gather lays down each chip's entire contribution before the next chip's, so the order is
    chip-major with groups nested inside — and that is what this reproduces.

    Getting it wrong is not a subtle numerical drift; the volume comes back permuted. The mesh
    parity test is the check.
    """
    tile_range = shard.tile_factor if shard is not None else 1
    row_range = shard.row_factor if shard is not None else 1

    emitted = []
    for row_chip in range(row_range):
        for tile_chip in range(tile_range):
            for rows, tile_factor, row_factor in padded_rows:
                local_tiles = rows.shape[0] // tile_factor
                local_queries = rows.shape[1] // row_factor
                # A group left unsplit on an axis sits on every chip along it, so the modulo
                # collapses to offset 0 and its rows are emitted once per chip.
                tile_start = (tile_chip % tile_factor) * local_tiles
                row_start = (row_chip % row_factor) * local_queries
                block = rows[tile_start : tile_start + local_tiles, row_start : row_start + local_queries]
                emitted.append(block.reshape(-1))
    return emitted


@dataclass
class NA3DDevicePlan:
    """Uploaded index tensors and masks for one ``(dims, kernel)`` on one mesh.

    Built once and cached: the index arithmetic is shape-dependent but weight-independent, so
    every block sharing a grid shape and kernel reuses this. When ``shard`` is set the indices
    are distributed and ``ccl_manager`` is the one that reassembles each group's output, so a
    plan carries everything its executor needs to know about the mesh.
    """

    plan: NA3DPlan
    groups: tuple[NA3DGroup, ...]
    restore_indices: ttnn.Tensor
    shard: NA3DShard | None = None
    ccl_manager: object | None = None


def build_device_plan(
    plan: NA3DPlan,
    *,
    mesh_device,
    dtype: ttnn.DataType = ttnn.bfloat16,
    ccl_manager=None,
    shard: NA3DShard | None = None,
) -> NA3DDevicePlan:
    """Upload a plan's gather indices and additive masks, optionally split across the mesh.

    Queries are gathered rather than sliced because a real grid has tens of thousands of
    tiles: as slices that is one op each, as a gather it is one op per *group*. The tiles of
    a group partition into the batch dimension, so a group becomes a single batched attention
    call sharing one mask.

    Sharding needs a ``ccl_manager`` to gather each group's output back, so without one the
    plan stays replicated however capable the mesh is — that keeps the single-chip parity tests
    on exactly the path they have always taken. Pass ``shard`` to override the default split.
    """
    if ccl_manager is None:
        shard = None
    elif shard is None:
        shard = NA3DShard.for_mesh(mesh_device)

    padded_rows: list[tuple[torch.Tensor, int, int]] = []
    groups = []
    for group in plan.groups:
        # (tiles, per_tile): ttnn.embedding maps a (batch, seq) index to (batch, seq, width),
        # which is the shape the attention call wants anyway, and a chunk of tiles is then a
        # slice of the *leading* dim. Reshaping these on device instead would change the
        # innermost dim, which is a data-movement kernel whose circular buffers overflow L1 at
        # a realistic key count.
        q_rows = torch.stack([_flat_indices(plan.dims, block) for block in group.query_slices])
        k_rows = torch.stack([_flat_indices(plan.dims, block) for block in group.key_slices])
        # bfloat16's most-negative value stands in for -inf: exp() of it underflows to zero,
        # which is what the masked softmax needs, and it survives the dtype round trip.
        mask = group_mask(group, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)

        tile_factor = shard.tile_factor if shard is not None else 1
        # A group with fewer query rows than the mesh axis is wide cannot be split along them.
        row_factor = shard.row_factor if shard is not None and group.n_queries >= shard.row_factor else 1

        q_rows = _pad_by_duplication(q_rows, dim=0, multiple=tile_factor)
        k_rows = _pad_by_duplication(k_rows, dim=0, multiple=tile_factor)
        q_rows = _pad_by_duplication(q_rows, dim=1, multiple=row_factor)
        mask = _pad_by_duplication(mask, dim=2, multiple=row_factor)
        padded_rows.append((q_rows, tile_factor, row_factor))

        tile_mesh_axis = shard.tile_axis if shard is not None and tile_factor > 1 else None
        row_mesh_axis = shard.row_axis if shard is not None and row_factor > 1 else None
        upload = {"device": mesh_device, "dtype": ttnn.uint32, "layout": ttnn.ROW_MAJOR_LAYOUT}
        groups.append(
            NA3DGroup(
                query_indices=from_torch(q_rows.to(torch.int32), mesh_axes=[tile_mesh_axis, row_mesh_axis], **upload),
                key_indices=from_torch(k_rows.to(torch.int32), mesh_axes=[tile_mesh_axis, None], **upload),
                mask=from_torch(
                    mask,
                    device=mesh_device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_axes=[None, None, row_mesh_axis, None],
                ),
                local_tiles=q_rows.shape[0] // tile_factor,
                local_queries=q_rows.shape[1] // row_factor,
                n_keys=group.n_keys,
                tile_factor=tile_factor,
                row_factor=row_factor,
            )
        )

    # Query tiles partition the volume, so the groups together visit every voxel at least once
    # — and more than once wherever padding duplicated a tile or a row. Inverting that mapping
    # puts the volume back together in one final gather; duplicates are free to collide because
    # they carry identical rows.
    order = torch.cat(_emitted_order(padded_rows, shard))
    volume = math.prod(plan.dims)
    covered = torch.zeros(volume, dtype=torch.bool)
    covered[order] = True
    assert covered.all(), f"plan covers {int(covered.sum())} of {volume} voxels"
    restore = torch.empty(volume, dtype=torch.int64)
    restore[order] = torch.arange(order.numel())

    return NA3DDevicePlan(
        plan=plan,
        groups=tuple(groups),
        restore_indices=ttnn.from_torch(
            restore.to(torch.int32).reshape(1, -1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        shard=shard,
        ccl_manager=ccl_manager,
    )


_GATHER_DEVICE_PLAN_CACHE: dict[tuple, NA3DDevicePlan] = {}


def cached_device_plan(
    dims: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    *,
    mesh_device,
    dtype: ttnn.DataType = ttnn.bfloat16,
    ccl_manager=None,
) -> NA3DDevicePlan:
    """Device plan for one geometry, built once per process.

    A plan depends only on the grid, the kernel, the dtype and how it is split — never on
    weights — but it uploads index tables and masks, so rebuilding it per block (every block of
    a stack shares a geometry) would dominate a decode. Keyed on the device and the CCL manager
    as well, since the uploaded tensors belong to one mesh and the split depends on the manager.
    """
    key = (tuple(dims), tuple(kernel_size), dtype, id(mesh_device), id(ccl_manager))
    plan = _GATHER_DEVICE_PLAN_CACHE.get(key)
    if plan is None:
        plan = build_device_plan(
            plan_na3d(tuple(dims), tuple(kernel_size)),
            mesh_device=mesh_device,
            dtype=dtype,
            ccl_manager=ccl_manager,
        )
        _GATHER_DEVICE_PLAN_CACHE[key] = plan
    return plan


# ---------------------------------------------------------------------------------------------
# The bricked plan: brick choice, query chunk, the C++ planner per shard, and the mask tables.
# ---------------------------------------------------------------------------------------------

# Plans depend on no weights but do upload an index table, so rebuilding one per block would
# dominate. Keyed on the geometry, exactly as the op's program cache is.
_BRICKED_PLAN_CACHE: dict = {}


def _window_origin(group_index, stride, window, volume, snap=0):
    """Host mirror of window_origin_on_axis. Only used to BUILD the interior mask, which the
    kernel then reads; the kernel keeps its own copy of the rule for boundary bricks."""
    first = group_index * stride
    last = min(first + stride - 1, volume - 1)
    centre = first + (last - first + 1) // 2  # NATTEN's leader: right of centre for an even group
    highest = volume - window
    origin = 0 if centre < window // 2 else min(centre - window // 2, highest)
    if snap <= 1:
        return origin
    lowest_containing = max(0, last + 1 - window)
    highest_containing = min(first, highest)
    down = (origin // snap) * snap
    if down >= lowest_containing:
        return down
    return down + snap if down + snap <= highest_containing else origin


def _site_in_brick(index, brick):
    per_time = brick[1] * brick[2]
    return (index // per_time, (index % per_time) // brick[2], (index % per_time) % brick[2])


def _regime_of(chunk_origin, chunk_extent, window, volume, stride, snap):
    """0 = every query in the chunk clamps low, 1 = none clamp, 2 = every query clamps high.

    Returns None when the chunk straddles a transition, where the pattern is not shared and the
    kernel must evaluate. Those are at most one chunk per edge per axis.

    Scanned over the CHUNK because the chunk is the unit that shares a window. Must agree with
    ``chunk_regime`` in the reader, which picks the uploaded set the same way.
    """
    origins = [
        _window_origin((chunk_origin + offset) // stride, stride, window, volume, snap)
        for offset in range(chunk_extent)
    ]
    highest = volume - window
    if all(origin == 0 for origin in origins):
        return 0
    if all(origin == highest for origin in origins):
        return 2
    centred = [
        _window_origin((chunk_origin + offset) // stride, stride, window, volume, snap) not in (0, highest)
        for offset in range(chunk_extent)
    ]
    return 1 if all(centred) else None


def _build_regime_masks(volume, context_window, stride, brick, chunk_bricks, plan):
    """``[1, 1, 32, 27 * gather_brick_count * 32]`` -- one mask set per (t, h, w) regime.

    A brick's mask depends on its position only through CLAMPING, and every fully-clamped brick
    on an axis shares the same window origin (0 low, volume-window high) and the same gather
    origin. So three classes per axis cover all but the transition bricks: 27 patterns instead
    of one per brick. This is the same collapse the reference gets from grouping query tiles by
    window geometry -- built here on the host and read by the kernel like K or V.
    """
    gather_bricks = plan["gather_bricks"]
    gather_brick_count = plan["gather_brick_count"]
    window = tuple(min(context_window[a], volume[a]) for a in range(3))
    # snap_extent_on_axis, in Python: legal wherever a whole brick lies inside one query group,
    # which is wherever the stride is a whole number of bricks.
    snap = tuple(brick[a] if stride[a] % brick[a] == 0 else 0 for a in range(3))

    # A representative CHUNK origin for each regime on each axis. The chunk is the unit that
    # shares a window, so it is the unit whose clamping decides which pattern applies.
    chunk_sites = tuple(chunk_bricks[axis] * brick[axis] for axis in range(3))
    representative = []
    for axis in range(3):
        found = {}
        for index in range(0, max(1, volume[axis] - chunk_sites[axis] + 1), chunk_sites[axis]):
            regime = _regime_of(index, chunk_sites[axis], window[axis], volume[axis], stride[axis], snap[axis])
            if regime is not None and regime not in found:
                found[regime] = index
        representative.append(found)

    tiles = gather_brick_count * SITES_PER_BRICK
    masks = torch.zeros(1, 1, SITES_PER_BRICK, 27 * tiles)

    for regime_time in range(3):
        for regime_height in range(3):
            for regime_width in range(3):
                regime = (regime_time * 3 + regime_height) * 3 + regime_width
                axes = (regime_time, regime_height, regime_width)
                if any(axes[a] not in representative[a] for a in range(3)):
                    continue  # this combination does not occur; leave it open, never selected
                base = tuple(representative[a][axes[a]] for a in range(3))
                gather_origin = tuple(
                    (_window_origin(base[a] // stride[a], stride[a], window[a], volume[a], snap[a]) // brick[a])
                    * brick[a]
                    for a in range(3)
                )
                for slot in range(gather_brick_count):
                    within = slot % (gather_bricks[1] * gather_bricks[2])
                    key_origin = (
                        gather_origin[0] + (slot // (gather_bricks[1] * gather_bricks[2])) * brick[0],
                        gather_origin[1] + (within // gather_bricks[2]) * brick[1],
                        gather_origin[2] + (within % gather_bricks[2]) * brick[2],
                    )
                    for row in range(SITES_PER_BRICK):
                        offset = _site_in_brick(row, brick)
                        query = tuple(base[a] + offset[a] for a in range(3))
                        low, high = [], []
                        for a in range(3):
                            origin = _window_origin(query[a] // stride[a], stride[a], window[a], volume[a], snap[a])
                            low.append(origin)
                            high.append(origin + window[a])
                        for column in range(SITES_PER_BRICK):
                            key_offset = _site_in_brick(column, brick)
                            key = tuple(key_origin[a] + key_offset[a] for a in range(3))
                            if not all(low[a] <= key[a] < high[a] for a in range(3)):
                                masks[0, 0, row, regime * tiles + slot * SITES_PER_BRICK + column] = float("-inf")
    return masks


def relative_mask_span(window_extent: int, brick_extent: int) -> tuple[int, int]:
    """Inclusive range of ``key_brick - query_brick`` a window can reach, on one axis.

    ``key_site - query_site`` must land in ``[-half, window - 1 - half]``, and each site is a
    brick offset plus a position inside the brick, so the relative BRICK offset is bounded by
    that range widened by ``brick - 1`` on both ends. 11 over a 2-brick gives [-3, 3]; over a
    4-brick, [-2, 2] -- 7 * 5 * 5 = 175 tiles for the whole plan.

    Transcribed in ``relative_mask_span`` in neighborhood_reader.cpp; the two MUST agree or the
    kernel indexes a tile the host never wrote.
    """
    half = window_extent // 2
    low = -((half + brick_extent - 1) // brick_extent)
    high = (window_extent - 1 - half + brick_extent - 1) // brick_extent
    return low, high


def _build_relative_masks(context_window, brick):
    """``[1, 1, 32, N * 32]`` indexed by the RELATIVE brick offset ``key_brick - query_brick``.

    At stride 1 an unclamped query centres its own window, so whether a key is visible depends
    only on ``key_site - query_site``. Both sites are a brick origin plus an offset within the
    brick, so the whole pattern is a function of the relative BRICK offset alone -- 175 tiles for
    an 11^3 window on a (2,4,4) brick, against the ~25M tiles the kernel generates per block.

    Indexing by the relative offset rather than the absolute gather slot is also what makes it
    CORRECT. ``gather_origin - chunk_origin`` is NOT constant: brick-aligning a clamped window
    origin shifts the phase, giving 75 distinct values at 1080p. A table keyed on the slot is
    therefore right only for chunks sharing the representative's phase and silently wrong for the
    rest, which is what put the uploaded table at PCC 0.914 against the torch reference. The
    relative offset has no such dependence.

    Boundary bricks -- those whose window clamps at a volume edge -- are NOT described here; the
    kernel keeps generating those, and there is at most one per edge per axis.
    """
    spans = [relative_mask_span(context_window[a], brick[a]) for a in range(3)]
    extents = [high - low + 1 for low, high in spans]
    half = [context_window[a] // 2 for a in range(3)]
    masks = torch.zeros(1, 1, SITES_PER_BRICK, extents[0] * extents[1] * extents[2] * SITES_PER_BRICK)

    for relative_time in range(spans[0][0], spans[0][1] + 1):
        for relative_height in range(spans[1][0], spans[1][1] + 1):
            for relative_width in range(spans[2][0], spans[2][1] + 1):
                relative = (relative_time, relative_height, relative_width)
                linear_brick_index = (
                    (relative_time - spans[0][0]) * extents[1] + (relative_height - spans[1][0])
                ) * extents[2] + (relative_width - spans[2][0])
                for row in range(SITES_PER_BRICK):
                    query_offset = _site_in_brick(row, brick)
                    for column in range(SITES_PER_BRICK):
                        key_offset = _site_in_brick(column, brick)
                        visible = True
                        for axis in range(3):
                            delta = relative[axis] * brick[axis] + key_offset[axis] - query_offset[axis]
                            if not (-half[axis] <= delta <= context_window[axis] - 1 - half[axis]):
                                visible = False
                                break
                        if not visible:
                            masks[0, 0, row, linear_brick_index * SITES_PER_BRICK + column] = float("-inf")
    return masks


def _query_chunk_bricks(stride: tuple[int, int, int], brick: tuple[int, int, int]) -> tuple[int, int, int]:
    """The largest chunk of bricks that still forms a single query group.

    A chunk is the set of queries sharing one gather. Making it bigger is the single largest
    lever in this op, because keys-gathered-per-query is ``gathered_box / chunk_queries`` and the
    box does NOT grow with the chunk so long as the chunk stays inside one query group -- at
    which point every row in the chunk has the same window. An 11^3 window costs 54 keys per
    query at one brick per chunk, and 4.8 at 5x2x2.

    So there is nothing to tune: one query group is exactly ``stride`` sites, and the chunk is
    that measured in bricks. A stride that is not a whole number of bricks on some axis simply
    cannot amortise along it, and gets one brick there.
    """
    # DIFFVAE_NA_CHUNK_BRICKS forces the chunk, decoupling it from the stride. Only meaningful
    # together with DIFFVAE_NA_UNSAFE_CHUNK=1, which lifts the plan's chunk==stride check: at
    # stride 1 the queries in a chunk do NOT share a window, so the broadcast mask is wrong and so
    # is the output. It exists to measure the ceiling -- 175 keys/query at chunk (1,1,1) against
    # 36 at (2,2,2) -- before paying for the per-brick mask that would make it correct.
    forced = os.environ.get("DIFFVAE_NA_CHUNK_BRICKS")
    if forced:
        return tuple(int(part) for part in forced.split(","))
    return tuple(
        stride_extent // brick_extent if stride_extent % brick_extent == 0 else 1
        for stride_extent, brick_extent in zip(stride, brick)
    )


def brick_override(volume: tuple[int, int, int]) -> tuple[int, int, int] | None:
    """``DIFFVAE_NA_BRICK``: force the brick instead of deriving it.

    Two forms. ``bt,bh,bw`` applies to EVERY volume -- fine while stage 5 was the only bricked
    caller, wrong once the deterministic stages brick too, since one brick cannot be forced on
    (21,68,120) and (84,272,480) at once. ``T,H,W:bt,bh,bw;T,H,W:bt,bh,bw`` is keyed by the FULL
    volume, so a per-stage A/B forces only the stage it names and every other stage keeps its
    derived brick. Returns None when nothing applies.
    """
    env = os.environ.get("DIFFVAE_NA_BRICK")
    if not env:
        return None
    if ":" not in env:
        return tuple(int(part) for part in env.split(","))
    for entry in env.split(";"):
        key, _, value = entry.partition(":")
        if tuple(int(part) for part in key.split(",")) == tuple(volume):
            return tuple(int(part) for part in value.split(","))
    return None


def cached_bricked_plan(volume, context_window, stride, brick, device, *, resident=None, shard_count=1, sp_axis=None):
    """Plan plus uploaded tables, cached per geometry. UNSHARDED IS THE ONE-SHARD CASE.

    ``resident`` is what one device HOLDS: its owned columns plus the halo its windows reach
    into. Omit it for an unsharded run -- resident becomes the volume, there is no halo, the
    query region is the whole volume, and the origin table replicates instead of sharding. The
    plan builder already models it that way: a ``shard_extent`` equal to the volume is not
    sharded, and a query region equal to the resident one is not a sub-region (see
    ``NeighborhoodConfig`` in neighborhood_plan.hpp), so nothing below needs a second path.

    One plan per shard, with the per-device gather tables stacked for a sharded upload. Every
    device runs the SAME program, so the plan's shapes -- chunk count, gathered bricks -- must
    agree across shards, and they do: the resident extent is uniform and every origin is
    brick-aligned, so only the origins themselves differ. Those ride the table, which is sharded
    over the mesh so each device reads its own. That is why only shard 0's plan is kept: it is
    the representative, and the assert below is what licenses treating it as one.
    """
    sharded = resident is not None
    resident = resident if sharded else volume
    query_chunk_bricks = _query_chunk_bricks(stride, brick)
    key = (volume, context_window, stride, brick, query_chunk_bricks, resident, shard_count, sp_axis, id(device))
    entry = _BRICKED_PLAN_CACHE.get(key)
    if entry is not None:
        return entry

    # Queries are the columns this shard OWNS; keys are those plus the halo. Telling the op the
    # difference is what stops it computing -- and Q from having to carry -- the halo's queries,
    # which belong to the neighbour and were discarded after every call. Unsharded there is no
    # halo, so the query region is the whole volume and this reduces to the identity.
    halo = halo_sites(min(context_window[2], volume[2]), brick[2]) if sharded else 0
    owned_width = resident[2] - 2 * halo
    query_extent = (resident[0], resident[1], owned_width)
    query_origin = (0, 0, halo)
    plans = []
    for shard_index in range(shard_count):
        # The device at the low edge sits BELOW the volume by one halo. Those columns are real
        # storage holding nothing the volume contains; no query owns them, no window reaches them.
        plans.append(
            ttnn.transformer.neighborhood_plan(
                volume,
                context_window,
                stride,
                brick,
                query_chunk_bricks=query_chunk_bricks,
                shard_extent=resident,
                shard_origin=(0, 0, shard_index * owned_width - halo),
                query_extent=query_extent,
                query_origin=query_origin,
            )
        )

    first = plans[0]
    for shard_index, plan in enumerate(plans[1:], start=1):
        for field in ("chunk_count", "gather_brick_count", "gather_bricks", "volume_chunks", "query_brick_count"):
            assert plan[field] == first[field], (
                f"shard {shard_index} plans a different {field} than shard 0 "
                f"({plan[field]} vs {first[field]}); one program cannot serve both"
            )

    stacked = torch.tensor([plan["gather_origin_table"] for plan in plans], dtype=torch.uint32).reshape(
        shard_count, 1, first["chunk_count"], first["gather_origin_columns"]
    )
    # sp_axis is None unsharded, which makes every placement a replicate -- the same distribution
    # a plain single-device upload produces, and the same (1, 1, chunks, columns) shape.
    first["gather_origin_tensor"] = from_torch(
        stacked,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_axes=[sp_axis, None, None, None],
    )
    first["query_chunk_bricks"] = query_chunk_bricks
    first["query_extent"] = query_extent
    first["query_origin"] = query_origin

    from loguru import logger

    logger.info(
        f"[neighborhood] {f'W-SHARDED x{shard_count}: ' if sharded else ''}volume={volume} "
        f"{f'resident={resident} ' if sharded else ''}"
        f"window={context_window} stride={stride} brick={brick} "
        f"chunk={query_chunk_bricks} bricks ({first['bricks_per_query_chunk'] * SITES_PER_BRICK} queries) "
        f"bricks={first['brick_count']} gather={first['gather_brick_count']} tiles "
        f"({first['gather_brick_count'] / first['bricks_per_query_chunk']:.2f} keys/query, waste "
        f"{first['gather_brick_count'] * SITES_PER_BRICK / (context_window[0] * context_window[1] * context_window[2]):.2f}x)"
    )
    # Uploading these is the single largest win in the op. Generating masks on device instead
    # costs 43.5 ms of a 53.8 ms block at stage-5 size -- 81% of the whole attention -- because
    # every gathered brick that straddles the window edge is evaluated per site, per chunk, every
    # block. There are only 27 distinct patterns, they depend on nothing but geometry, and this
    # builds them once and keeps them resident.
    # At stride 1 the regime table above does not apply: its 27 patterns describe chunks that
    # share ONE window, which is a GNA property. Every query centres its own window here, so the
    # pattern is a function of the relative brick offset instead -- see _build_relative_masks.
    first["relative_mask"] = stride == (1, 1, 1)
    if first["relative_mask"]:
        # The RELATIVE table depends on nothing but the window and the brick, so it uploads once
        # and serves every shard.
        masks = _build_relative_masks(context_window, brick)
    elif sharded:
        # The REGIME sets cannot be uploaded once sharded: they are enumerated against a single
        # shard origin and every shard has its own, so the sharded path generates every tile on
        # device.
        masks = None
    else:
        masks = _build_regime_masks(volume, context_window, stride, brick, query_chunk_bricks, first)
    first["interior_mask_tensor"] = (
        None if masks is None else ttnn.from_torch(masks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    )
    _BRICKED_PLAN_CACHE[key] = first
    return first


def _tiles_per_kv_chunk(gather_brick_count: int) -> int:
    """Largest chunk that fits DST and divides the gather evenly.

    A chunk's score tiles stay live in the destination registers through the row max and the
    exp, so a chunk wider than DST silently returns wrong numbers rather than faulting. Eight is
    the same bound that makes the rest of the SDPA family use ``k_chunk_size = 256``.
    """
    DST_CAPACITY_TILES = 8
    # DIFFVAE_NA_KV_CHUNK_TILES forces a chunk width. 8 tiles = 256 tokens, the k_chunk_size the
    # rest of the SDPA family uses; it need not divide the gather because the ragged tail is
    # padded with fully-masked slots. Fewer, wider chunks means fewer flash iterations.
    forced = os.environ.get("DIFFVAE_NA_KV_CHUNK_TILES")
    if forced:
        return max(1, min(int(forced), DST_CAPACITY_TILES, gather_brick_count))
    for candidate in range(min(gather_brick_count, DST_CAPACITY_TILES), 0, -1):
        if gather_brick_count % candidate == 0:
            return candidate
    return 1


def halo_sites(context_window_extent: int, brick_extent: int) -> int:
    """Sites of a neighbour's data a shard needs on each side of one axis.

    The window reaches ``context_window // 2`` past a query, and the exchange moves whole bricks
    because that is the unit the op addresses.
    """
    reach = context_window_extent // 2
    return -(-reach // brick_extent) * brick_extent  # ceil, in whole bricks


_BRICK_CHOICE_CACHE: dict = {}


def _choose_sharded_brick(volume, context_window, stride, width_local, shard_count):
    """The 32-site brick that makes the GATHER smallest, measured in bricks by the real planner.

    ``neighborhood_choose_brick`` minimises the window union in SITES, and at stride 1 that is the
    wrong objective: a query brick's score tiles, its K/V reads and its mask tiles are all one per
    gathered BRICK, and brick alignment inflates that. A stride-1 window origin sits at
    ``-(window // 2) mod brick`` off a brick boundary on every axis, so the two objectives rank
    differently -- at 1080p (2,4,4) has the smaller union in sites (2352 against 2592) and the
    larger gather in bricks (175 against 147).

    Asked of the planner rather than derived, because the count depends on the worst misalignment
    over every chunk on every shard, which is what ``build_plan`` measures and what a hand formula
    got wrong for (2,2,8) -- 147 on paper, 196 in the plan.

    Only for stride 1. Where the stride is a whole number of bricks the window origin snaps to a
    brick boundary, nothing is misaligned and the two objectives agree.
    """
    if stride != (1, 1, 1):
        return tuple(ttnn.transformer.neighborhood_choose_brick(context_window))

    key = (volume, context_window, stride, width_local, shard_count)
    cached = _BRICK_CHOICE_CACHE.get(key)
    if cached is not None:
        return cached

    default = tuple(ttnn.transformer.neighborhood_choose_brick(context_window))
    best, best_gather, best_query = default, None, None
    for brick_time in range(1, SITES_PER_BRICK + 1):
        for brick_height in range(1, SITES_PER_BRICK + 1):
            if SITES_PER_BRICK % (brick_time * brick_height):
                continue
            brick_width = SITES_PER_BRICK // (brick_time * brick_height)
            if brick_time * brick_height * brick_width != SITES_PER_BRICK:
                continue
            # Odd widths are legal. They were excluded while the K/V halo moved in NATURAL order,
            # where an odd halo could not fold W columns into the stick and `neighbor_pad` hung at
            # a 128 B stick. Every path now exchanges in bricked order (stick 32 * channels), so
            # nothing depends on the halo's parity -- and width 1 is the ONLY brick the planner
            # accepts at the deterministic stages' W_local = 15: shard origins must be brick-aligned,
            # which reduces to brick_width | width_local, and 15 has no even divisor.
            brick = (brick_time, brick_height, brick_width)
            # A brick deeper than the volume on any axis is degenerate: it pads that axis out past
            # its own extent, so every brick is mostly ghost sites and the axis contributes a single
            # slot to the gather -- which this objective then scores as excellent. On a 12-frame
            # volume the search picked (16, 1, 2), gathering 77 bricks against the 147 a real brick
            # needs, and the op wedged. No effect at 1080p, where (2, 8, 2) is well inside
            # (84, 272, 480); this only rules out choices that were never meaningful.
            if any(extent > limit for extent, limit in zip(brick, volume)):
                continue
            halo = halo_sites(min(context_window[2], volume[2]), brick_width)
            if halo > width_local:
                continue
            resident = (volume[0], volume[1], width_local + 2 * halo)
            try:
                plans = [
                    ttnn.transformer.neighborhood_plan(
                        volume,
                        context_window,
                        stride,
                        brick,
                        query_chunk_bricks=_query_chunk_bricks(stride, brick),
                        shard_extent=resident,
                        shard_origin=(0, 0, index * width_local - halo),
                        # The owned bricks only, as cached_bricked_plan passes: the planner refuses a query
                        # region that starts below the volume, which shard 0's halo does, and the
                        # except below would otherwise discard EVERY candidate and fall back to the
                        # default brick (200 gathered bricks at 1080p against 168).
                        query_extent=(volume[0], volume[1], width_local),
                        query_origin=(0, 0, halo),
                    )
                    for index in range(shard_count)
                ]
            except (ValueError, RuntimeError):
                continue  # a brick the planner refuses for this geometry
            if any(plan["gather_brick_count"] != plans[0]["gather_brick_count"] for plan in plans):
                continue  # one program cannot serve shards that gather differently
            gather = plans[0]["gather_brick_count"]
            query_bricks = plans[0]["query_brick_count"]
            # Tie-breaks, in order: fewest gathered bricks, then smallest halo, then the deepest
            # brick in TIME. A smaller halo is less to exchange, brick-permute and drop, and that
            # one is reasoned. Preferring depth in time is MEASURED and not explained -- shapes
            # that gather identically and carry the same halo do not run at the same speed, and
            # the deeper time extent won. It is a preference, not a rule: re-measure it if the
            # volume, window or shard count changes.
            # Not scored: the query brick count (ghost padding). A total-work objective
            # (query x gather) would flip the 1080p stage-5 pick from (8,2,2) to (2,8,2) against
            # its measured 11 %/brick advantage, so the padding is logged for the reader instead.
            score = (gather, halo, -brick_time)
            if best_gather is None or score < best_gather:
                best, best_gather, best_query = brick, score, query_bricks

    from loguru import logger

    if best_gather is None:
        logger.info(f"[neighborhood] no candidate brick plans at width_local={width_local}; using {default}")
    else:
        logger.info(
            f"[neighborhood] brick {best} gathers {best_gather[0]} bricks over {best_query} query bricks "
            f"({best_gather[0] * best_query} tile pairs per shard, halo {best_gather[1]}; "
            f"choose_brick would pick {default})"
        )
    _BRICK_CHOICE_CACHE[key] = best
    return best
