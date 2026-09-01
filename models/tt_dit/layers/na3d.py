# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""3D neighborhood attention, as dense masked attention over grouped query tiles.

Every block of the LTX-2.5 DiffVAE decoder attends over a local 3D window rather than the
whole volume (``(3,7,7)`` and ``(3,5,5)`` in the deterministic stages, ``(11,11,11)`` in the
diffusion stage). Upstream runs NATTEN's fused CUDA kernel; there is no such primitive here,
so we use the identity their own natten-free fallback relies on: a tile of queries attends to
a bounded span of keys, so it can be evaluated as ordinary attention with an additive mask.

The window rule is the part worth stating precisely, because it is not the usual sliding
window. NATTEN keeps the window *size* constant and shifts it inward at the boundaries, so
every query attends to exactly ``K`` keys along each axis — a query at index 0 attends to
``[0, K)``, not to a truncated ``[0, K//2]``. A truncating window would look plausible and be
wrong everywhere near an edge.

The plan is separated from execution because the index arithmetic is the hard part and is
shared: :func:`plan_na3d` is pure Python, :func:`na3d_torch` executes it on host (and is what
the parity test holds against upstream), and the ttnn executor consumes the same plan.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from dataclasses import dataclass

import torch
from loguru import logger

import ttnn

from ..utils import decode_tree
from ..utils.tensor import from_torch

#: Sub-profile of the W-sharded attention (K/V all-gather vs fused SDPA vs head all-gather), summed
#: across every call in a diff-step. Gated by DIFFVAE_STAGE_TIMING; stage 5 clears + reports it.
SP_W_PROF: dict[str, float] = {}
#: Shared with diffvae_ltx_stage5 so the two cannot disagree about whether timing is on.
_SP_W_TIMING = decode_tree.ENABLED


@contextlib.contextmanager
def _sp_w_prof(mesh, key: str, *, category: str | None = None):
    """Accumulate into SP_W_PROF as before, and record the span in the decode tree.

    The tree attribution needs nothing passed down from the model: open_span reads a thread-local
    stack, so this lands under whichever stage or diff block is currently open.
    """
    if not _SP_W_TIMING:
        yield
        return
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    span = decode_tree.open_span(key, category=category)
    try:
        yield
    except BaseException:
        decode_tree.abort_span(span)
        raise
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) * 1000
    decode_tree.close_span(span, ms)
    SP_W_PROF[key] = SP_W_PROF.get(key, 0.0) + ms


# Cap on one tile group's [Nq, Nk] score block. Bounds both the additive mask allocation and
# the score materialization; the tile search shrinks axes until the product fits.
DEFAULT_SCORE_BUDGET = 2**22

# Cap on the elements gathered for one attention call's K and V together. Peak device memory is
# what this bounds, and it is separate from the score budget above, which bounds a *tile's*
# window: a group can hold tens of thousands of tiles, and running them as one call is what
# scales with resolution. 2**29 elements is 1 GB in bfloat16 for K and V combined.
#
# The bound is per chip, so a sharded plan (see :class:`NA3DShard`) splits a group's tiles
# before this applies and needs proportionally fewer chunks for the same grid.
DEFAULT_CHUNK_BUDGET = 2**29


def window_bounds(length: int, kernel: int, stride: int = 1) -> tuple[list[int], list[int]]:
    """Per-index ``(start, end)`` of the attended window along one axis.

    Implements NATTEN's constant-size inward-shifted window: the start is the query index
    less half the kernel, clamped so the window never leaves ``[0, length)``. When the axis
    is shorter than the kernel every query attends to the whole axis.

    ``stride`` is GNA's query-group size: runs of ``stride`` queries share the window of their
    center-most member, biased right for even groups so the bias opposes the inward shift's.
    ``stride=1`` is standard neighborhood attention, every query centered on its own window.
    The C++ twin is ``nbr_shift_start`` in windowed_loop_geometry.hpp; they must agree exactly.
    """
    kernel = min(kernel, length)
    half = kernel // 2
    last_start = length - kernel
    leaders = [min((i // stride) * stride + stride // 2, length - 1) for i in range(length)]
    starts = [min(max(q - half, 0), last_start) for q in leaders]
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

    @property
    def waste_factor(self) -> float:
        """Dense score elements evaluated per element an exact NA kernel would evaluate.

        1.0 is a perfect kernel; the dense formulation always pays more because a tile's key
        span covers the union of its queries' windows. Useful for choosing tile sizes.
        """
        ideal = math.prod(self.dims) * math.prod(min(k, d) for k, d in zip(self.kernels, self.dims))
        dense = sum(len(g.query_slices) * g.n_queries * g.n_keys for g in self.groups)
        return dense / ideal


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


_PLAN_CACHE: dict[tuple, NA3DDevicePlan] = {}


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
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = build_device_plan(
            plan_na3d(tuple(dims), tuple(kernel_size)),
            mesh_device=mesh_device,
            dtype=dtype,
            ccl_manager=ccl_manager,
        )
        _PLAN_CACHE[key] = plan
    return plan


def _gather_stack(local: ttnn.Tensor, device_plan: NA3DDevicePlan) -> ttnn.Tensor:
    """Gather every chip's ``(rows, width)`` contribution into the full stack on every chip.

    Both gathers are along dim 0, one per mesh axis, and they run in the order
    :func:`_emitted_order` assumes: the tile axis first, then the row axis.
    """
    shard = device_plan.shard
    if shard is None:
        return local

    for mesh_axis in (shard.tile_axis, shard.row_axis):
        gathered = device_plan.ccl_manager.all_gather(local, dim=0, mesh_axis=mesh_axis, use_hyperparams=False)
        ttnn.deallocate(local)
        local = gathered
    return local


def neighborhood_attention_3d_op(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    scale: float | None = None,
) -> ttnn.Tensor:
    """3D neighborhood attention via the SDPA op's on-device ``neighborhood_3d`` mask.

    Same contract as :func:`neighborhood_attention_3d` — ``q``/``k``/``v`` are
    ``(B, T, H, W, num_heads, head_dim)`` and the return is ``(B, T, H, W, num_heads*head_dim)``
    in ROW_MAJOR — but instead of gathering each window and running dense masked attention it
    flattens the volume to a single ``(B, num_heads, T*H*W, head_dim)`` sequence and lets the op
    synthesize the ``(kt, kh, kw)`` inward-shifted neighborhood mask on device. One op call, no
    per-group gather, no mask upload, no CCL — so peak memory never sees the tens-of-GB K/V the
    grouped path gathers at full resolution.

    The op currently leaves the K-range full (step 2 of the generalization), so this streams all
    K per query chunk and its compute is O(S^2); step 3 narrows the K-range to the neighborhood.
    Correctness is independent of that: the mask makes out-of-window scores -inf either way.

    Single-mesh / replicated only for now — sharding over T (SP) rides on the op's windowed
    K-range work, not this gather-free path, so ``device_plan``/``ccl_manager`` do not apply here.
    """
    batch, t, h, w, heads, head_dim = tuple(q.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"
    if scale is None:
        scale = head_dim**-0.5
    kernels = tuple(min(kk, d) for kk, d in zip(kernel_size, (t, h, w)))
    seq = t * h * w

    # (B, T, H, W, NH, HD) -> (B, NH, S, HD): the op treats (B, NH) as batch dims and attends over
    # S = T*H*W (flattened T-outer, matching the mask's grid convention). Merging T,H,W and
    # splitting off heads are pure stride changes in ROW_MAJOR; SDPA's matmuls want TILE.
    def to_seq(x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch, seq, heads, head_dim))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    tq, tk, tv = (to_seq(x) for x in (q, k, v))
    # TEMP (phase-2 lever test): raise the SDPA chunk size to cut per-32-chunk iteration overhead
    # (default q/k chunk = 32 => Sk_chunk_t=1 => thousands of tiny masked matmuls over the box).
    prog_config = None
    _qc = os.environ.get("DIFFVAE_SDPA_QCHUNK")
    _kc = os.environ.get("DIFFVAE_SDPA_KCHUNK")
    if _qc or _kc:
        grid = q.device().compute_with_storage_grid_size()
        prog_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False,
            q_chunk_size=int(_qc) if _qc else 32,
            k_chunk_size=int(_kc) if _kc else 32,
        )
    attended = ttnn.transformer.scaled_dot_product_attention(
        tq, tk, tv, is_causal=False, neighborhood_3d=(t, h, w, *kernels), scale=scale, program_config=prog_config
    )
    for tensor in (tq, tk, tv):
        ttnn.deallocate(tensor)

    # (B, NH, S, HD) -> (B, T, H, W, NH*HD) ROW_MAJOR.
    attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
    attended = ttnn.permute(attended, (0, 2, 1, 3))
    return ttnn.reshape(attended, (batch, t, h, w, heads * head_dim))


def neighborhood_attention_3d_op_fused(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    scale: float | None = None,
) -> ttnn.Tensor:
    """Fused-gather variant of :func:`neighborhood_attention_3d_op` (build-out in progress).

    Same flatten-to-(B, NH, S, HD) contract as the ``op`` backend, but Q stays TILE while K/V are
    handed to the op in ROW_MAJOR so the SDPA reader can densely gather each query chunk's window
    rows (row-granular) into a contiguous cb_k/cb_v -- dense flash over only real window tokens, no
    scattered active-tile streaming. Selected with ``neighborhood_gather=True``.
    """
    batch, t, h, w, heads, head_dim = tuple(q.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"
    if scale is None:
        scale = head_dim**-0.5
    kernels = tuple(min(kk, d) for kk, d in zip(kernel_size, (t, h, w)))
    seq = t * h * w

    # (B, T, H, W, NH, HD) -> (B, NH, S, HD). Q -> TILE (matmul input); K/V stay ROW_MAJOR (the op
    # gathers their window rows on device, so it wants row-granular sticks, not tiles).
    def to_seq(x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch, seq, heads, head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))  # (B, NH, S, HD), ROW_MAJOR

    tq = ttnn.to_layout(to_seq(q), ttnn.TILE_LAYOUT)
    # K/V ROW_MAJOR, then reshape (B, NH, S, HD) -> (B, NH, T*H, W*HD): a pure contiguous reshape that
    # makes each DRAM page a full W-row, so the reader gathers a whole (t,h) box-row in one coalesced
    # read (W-run coalescing) instead of one tiny read per token.
    tk, tv = (ttnn.reshape(to_seq(x), (batch, heads, t * h, w * head_dim)) for x in (k, v))
    # Pin the SDPA chunk sizes so the host-precomputed mask layout matches the device exactly.
    # DIFFVAE_SDPA_KCHUNK/QCHUNK sweep them (a larger k_chunk_size amortizes softmax over more of the
    # densely packed box per flash step); default 32 => Sq_chunk_t = Sk_chunk_t = 1.
    q_chunk_size = int(os.environ.get("DIFFVAE_SDPA_QCHUNK", 32))
    k_chunk_size = int(os.environ.get("DIFFVAE_SDPA_KCHUNK", 32))
    grid = q.device().compute_with_storage_grid_size()
    prog_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )
    attended = ttnn.transformer.scaled_dot_product_attention(
        tq,
        tk,
        tv,
        is_causal=False,
        neighborhood_3d=(t, h, w, *kernels),
        neighborhood_gather=True,
        scale=scale,
        program_config=prog_config,
    )
    for tensor in (tq, tk, tv):
        ttnn.deallocate(tensor)

    # (B, NH, S, HD) -> (B, T, H, W, NH*HD) ROW_MAJOR.
    attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
    attended = ttnn.permute(attended, (0, 2, 1, 3))
    return ttnn.reshape(attended, (batch, t, h, w, heads * head_dim))


def neighborhood_attention_3d_op_sp(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    sp_axis: int,
    ccl_manager,
    scale: float | None = None,
) -> ttnn.Tensor:
    """SP-over-T drop-in for :func:`neighborhood_attention_3d_op`: same replicated in/out contract
    (``q``/``k``/``v`` full ``(B, T, H, W, NH, HD)`` on every chip, returns full
    ``(B, T, H, W, NH*HD)``), but the attention compute is split ``sp`` ways over the temporal axis.

    Q is partitioned over T across ``sp_axis`` (``ttnn.mesh_partition`` -- a per-device slice, the
    inverse of all_gather), so each chip attends only its ``T/sp`` frames; K/V stay full/replicated
    (the T window reaches a few frames past a shard edge, so replicating K/V avoids a halo). Each
    chip is told its global frame origin through a per-device offset tensor, and the sharded outputs
    are all-gathered back along T, so every chip ends with the full volume -- bit-identical (mod
    bf16) to the replicated executor, at 1/sp of the attention work per chip. Pointwise work in the
    block stays replicated; only the attention is parallelized.

    Requires whole-frame shards (T divisible by ``mesh.shape[sp_axis]``) and a tile-aligned shard
    origin (``(T/sp) * H * W`` a multiple of TILE_HEIGHT); the large stages satisfy both.
    """
    mesh = q.device()
    sp = int(list(mesh.shape)[sp_axis])
    batch, t, h, w, heads, head_dim = tuple(q.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"
    hw = h * w
    seq_full = t * hw
    width = heads * head_dim
    if sp == 1:  # nothing to split -- fall back to the plain replicated op path
        return neighborhood_attention_3d_op(q, k, v, kernel_size=kernel_size, scale=scale)
    assert t % sp == 0, f"T={t} must split evenly over sp={sp}"
    seq_local = seq_full // sp
    tile_height = 32
    assert seq_local % tile_height == 0, f"shard origin (T/sp)*H*W={seq_local} must be a multiple of {tile_height}"
    if scale is None:
        scale = head_dim**-0.5
    kernels = tuple(min(kk, d) for kk, d in zip(kernel_size, (t, h, w)))

    def to_seq(x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch, seq_full, heads, head_dim))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    tk = to_seq(k)
    tv = to_seq(v)
    # (B, NH, S, HD) full -> this chip's frame-shard (B, NH, S/sp, HD). mesh_partition slices S/sp per
    # device along sp_axis at S/sp boundaries = whole frames (T divisible by sp), so no fabric traffic.
    tq = ttnn.mesh_partition(to_seq(q), dim=2, cluster_axis=sp_axis)

    # One offset per chip along sp_axis: chip at position p holds seq [p*seq_local, ...) -- the same
    # slice mesh_partition assigns. Distribute the (sp,) table so page 0 per chip is its own origin.
    off_tt = ccl_manager.get_shard_offsets(sp, seq_local, sp_axis)

    attended = ttnn.transformer.scaled_dot_product_attention(
        tq,
        tk,
        tv,
        is_causal=False,
        neighborhood_3d=(t, h, w, *kernels),
        scale=scale,
        windowed_q_token_offset=0,
        windowed_q_token_offset_tensor=off_tt,
    )
    for tensor in (tq, tk, tv):
        ttnn.deallocate(tensor)

    # (B, NH, S/sp, HD) -> (B, S/sp, width) local, all-gather along seq across sp_axis (chip order is
    # frame order, so no reshuffle) -> full (B, T, H, W, width) on every chip.
    attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
    attended = ttnn.permute(attended, (0, 2, 1, 3))
    local = ttnn.reshape(attended, (batch, seq_local, width))
    full = ccl_manager.all_gather(local, dim=1, mesh_axis=sp_axis, use_hyperparams=False)
    return ttnn.reshape(full, (batch, t, h, w, width))


def neighborhood_attention_3d_op_sp_w(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    sp_axis: int,
    ccl_manager,
    scale: float | None = None,
) -> ttnn.Tensor:
    """Spatial sequence-parallelism over W. Same replicated in/out contract as
    :func:`neighborhood_attention_3d_op` (q/k/v full ``(B, T, H, W, NH, HD)``, returns full
    ``(B, T, H, W, NH*HD)``), but the attention compute is split ``sp`` ways over the W axis.

    Mirrors :func:`neighborhood_attention_3d_op_sp` exactly -- shard Q, keep K/V full/replicated,
    one all_gather, no halo exchange -- so it reuses that path's deadlock-free CCL pattern rather
    than the ``neighbor_pad`` + ``all_gather`` chain (which hangs the fabric). The only twist is that
    W is the *inner* spatial axis, so a flat-sequence slice is not a W-band. The volume is therefore
    permuted to W-outer ``(B, W, T, H, ...)`` before flattening, which makes each chip's
    ``ttnn.mesh_partition`` slice a contiguous W-band; the neighborhood mask is axis-order-agnostic,
    so it is simply told its grid as ``(W, T, H)`` with the matching ``(kw, kt, kh)`` kernel and each
    chip's flattened W-outer origin. Outputs are all-gathered along the sequence (chip order = W
    order) and permuted back to ``(B, T, H, W, width)``.

    Costs full K/V per chip (like the T-shard path), trading the halo path's ~1/sp K/V memory for a
    known-good CCL sequence. Requires W divisible by ``mesh.shape[sp_axis]`` and a tile-aligned shard
    origin (``(W/sp) * T * H`` a multiple of ``TILE_HEIGHT``).
    """
    mesh = q.device()
    sp = int(list(mesh.shape)[sp_axis])
    batch, t, h, w, heads, head_dim = tuple(q.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"
    width = heads * head_dim
    if sp == 1:  # nothing to split -- fall back to the plain replicated op path
        return neighborhood_attention_3d_op(q, k, v, kernel_size=kernel_size, scale=scale)
    assert w % sp == 0, f"W={w} must split evenly over sp={sp}"
    seq_full = w * t * h  # W-outer flatten
    seq_local = seq_full // sp
    tile_height = 32
    assert seq_local % tile_height == 0, f"shard origin (W/sp)*T*H={seq_local} must be a multiple of {tile_height}"
    if scale is None:
        scale = head_dim**-0.5
    kt, kh, kw = (min(kk, d) for kk, d in zip(kernel_size, (t, h, w)))

    # Flatten W-outer so a contiguous sequence slice is a W-band; heads are merged then re-split so the
    # spatial reorder is a single 5D permute (heads kept out of it). The mask is axis-order-agnostic,
    # so it later reads the grid as (W, T, H) -- see the neighborhood_3d argument below.
    def to_seq(x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch, t, h, w, width))
        x = ttnn.permute(x, (0, 3, 1, 2, 4))  # (B, W, T, H, width)
        x = ttnn.reshape(x, (batch, seq_full, heads, head_dim))
        x = ttnn.permute(x, (0, 2, 1, 3))  # (B, NH, S, HD)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    tk = to_seq(k)
    tv = to_seq(v)
    # (B, NH, S, HD) full -> this chip's W-band (B, NH, S/sp, HD). mesh_partition slices S/sp per
    # device along sp_axis at S/sp boundaries = whole W-columns (W divisible by sp), so no fabric.
    tq = ttnn.mesh_partition(to_seq(q), dim=2, cluster_axis=sp_axis)

    # One flattened W-outer origin per chip along sp_axis: chip at position p holds seq
    # [p*seq_local, ...), the same slice mesh_partition assigns. The mask adds it to each local query
    # index to recover the query's global (w, t, h).
    off_tt = ccl_manager.get_shard_offsets(sp, seq_local, sp_axis)

    attended = ttnn.transformer.scaled_dot_product_attention(
        tq,
        tk,
        tv,
        is_causal=False,
        neighborhood_3d=(w, t, h, kw, kt, kh),  # grid given W-outer to match the flatten
        scale=scale,
        windowed_q_token_offset=0,
        windowed_q_token_offset_tensor=off_tt,
    )
    for tensor in (tq, tk, tv):
        ttnn.deallocate(tensor)

    # (B, NH, S/sp, HD) -> (B, S/sp, width) local, all-gather along seq across sp_axis (chip order is
    # W order, so no reshuffle) -> full W-outer (B, W, T, H, width), permuted back to (B, T, H, W, width).
    attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
    attended = ttnn.permute(attended, (0, 2, 1, 3))
    local = ttnn.reshape(attended, (batch, seq_local, width))
    full = ccl_manager.all_gather(local, dim=1, mesh_axis=sp_axis, use_hyperparams=False)
    full = ttnn.reshape(full, (batch, w, t, h, width))
    return ttnn.permute(full, (0, 2, 3, 1, 4))  # (B, T, H, W, width)


def _pick_block(t_full: int, h_full: int, w_local: int, kmax: int = 11, gna: bool = False):
    """Largest tile-legal (bt,bh,bw) block for this shard's (T, H, w_local): each dim divides its axis,
    block_vol is a multiple of 32 and in [128, 512]. Ties broken by the smallest neighborhood box. Returns
    None if no legal block exists (caller falls back to the strided path). See box_model.py / Phase 0.

    ``gna`` picks for a GNA stride equal to the block instead. That inverts the objective: the box is then
    the kernel on every axis no matter how the block is shaped, so the box term is constant and volume --
    which sets the chunk count -- becomes the whole objective. It also caps each block dim at ``kmax``,
    since a stride above its kernel is rejected host-side (the group would outgrow the window it shares).

    This objective is purely a speed objective and it is not free. It optimizes toward block dims AT
    ``kmax``, i.e. stride == kernel, which is the largest window displacement a group can have. MEASURED
    at the 1080p stage-5 grid, block (11,4,8) vs stride-1 attention on the same inputs: PCC 0.51 on iid
    Q/K/V, 0.72 at a spatial correlation length of 8 tokens; fused-sdpa 6.8x faster (181->27 ms), whole
    block 1.95x. T carries almost all of the error (stride_t alone is PCC 0.72/0.85) because 11 == the
    kernel; (1,2,2) holds 0.97. A network trained at stride 1 cannot absorb the picked block -- constrain
    the stride, or leave GNA off, unless retraining.
    """

    def divs(n):
        return [d for d in range(1, n + 1) if n % d == 0]

    best = None
    for bt in divs(t_full):
        for bh in divs(h_full):
            for bw in divs(w_local):
                vol = bt * bh * bw
                if vol % 32 or not (128 <= vol <= 512):
                    continue
                if gna and max(bt, bh, bw) > kmax:
                    continue
                box = (bt + kmax - 1) * (bh + kmax - 1) * (bw + kmax - 1)
                # MEASURED (6s sweep, 2026-08-18): fused-sdpa cost tracks the BOX, not q_chunk/vol -- the
                # box's outer-axis (W,H) extent sets how far apart the reader's k-segments are, so a small
                # box beats a large-vol block even at 3x the chunk count (the block reorder is per-call, not
                # per-chunk). Minimize box, tie-break LARGER vol (fewer chunks). (5,8,4) over (5,8,12): fused
                # 10.2s->7.6s, 6s decode 25.0s->~22s. Old key (-vol, box) picked the slow large-vol block.
                key = (-vol, box) if gna else (box, -vol)
                if best is None or key < best[0]:
                    best = (key, (bt, bh, bw))
    return best[1] if best else None


def _deep_prof(mesh, key: str, *, category: str | None = None):
    """_sp_w_prof, but only under DIFFVAE_BLOCK_PROF -- see decode_tree.DEEP."""
    if not decode_tree.DEEP:
        return contextlib.nullcontext()
    return _sp_w_prof(mesh, key, category=category)


def neighborhood_attention_3d_op_sp_w_sharded(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    dims: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    sp_axis: int,
    ccl_manager,
    scale: float | None = None,
    tp_axis: int | None = None,
    heads_presharded: bool = False,
    flat_seq: bool = False,
    gna_stride: tuple[int, int, int] | None = None,
) -> ttnn.Tensor:
    """SP-over-W with SHARDED input AND output, for full-stage spatial parallelism.

    The W analog of :func:`neighborhood_attention_3d_op_sp_sharded`: ``q``/``k``/``v`` are THIS
    chip's contiguous W-slice ``(B, T, H, W/sp, NH, HD)`` and the return is this chip's W-slice of
    the output ``(B, T, H, W/sp, NH*HD)``. K/V are all-gathered to the full W internally (a W window
    reaches a few columns past a shard edge), but Q and the output stay W-sharded so the block's
    residual adds and the surrounding pointwise ops all stay 1/sp in both compute and memory.
    ``dims`` is the FULL ``(T, H, W)``.

    W is the inner spatial axis, so the volume is flattened W-outer (as in
    :func:`neighborhood_attention_3d_op_sp_w`): a contiguous sequence is then a W-band and the mask
    is told its grid as ``(W, T, H)`` with the matching ``(kw, kt, kh)`` kernel. Each chip is told
    its flattened W-outer origin. Requires W divisible by the mesh axis and a tile-aligned shard
    origin (``(W/sp) * T * H`` a multiple of ``TILE_HEIGHT``).

    ``tp_axis`` (optional) adds TENSOR PARALLELISM OVER HEADS on a second, orthogonal mesh axis:
    attention is independent per head, so each chip along ``tp_axis`` keeps only ``heads/tp`` of the
    heads. The head split is a communication-free per-device slice of the (replicated-over-tp) q/k/v
    done BEFORE the K/V all-gather -- so the gather that is the memory wall shrinks by ``tp`` too --
    and the heads are all-gathered back over ``tp_axis`` right after the flash, so the output
    projection and residual downstream see the full width unchanged. Composes with the W-shard: the
    two all-gathers are over different mesh axes.
    """
    mesh = q.device()
    sp = int(list(mesh.shape)[sp_axis])
    t_full, h_full, w_full = dims
    if flat_seq:
        # q/k/v arrive as the (B, NH, S, HD) TILE that nlp_create_qkv_heads emits, S in (t,h,w)
        # order, and the result comes back as (tokens, NH*HD). Building the 6-D volume the other
        # backends take costs a permute in and two more out for a reorder of S alone.
        assert heads_presharded, "flat_seq expects the caller's heads to be presharded"
        batch, heads, _seq_in, head_dim = tuple(q.shape)
        t, h, w_local = t_full, h_full, w_full // sp
    else:
        batch, t, h, w_local, heads, head_dim = tuple(q.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"
    assert (t, h) == (t_full, h_full), f"q dims {(t, h)} != full {(t_full, h_full)}"
    assert w_local * sp == w_full, f"W={w_full} must split evenly over sp={sp} (got W_local={w_local})"

    # TP-over-heads: partition q/k/v on the head axis across tp_axis. This is a pure per-device slice
    # (each chip selects its head band from the replicated tensor -- no comms), done here before the
    # heads are folded into the sequence so the K/V all-gather below moves only this chip's heads.
    # heads_presharded=True: the caller (column-parallel qkv) already emitted only this chip's heads,
    # so skip the slice -- ``heads`` is already local; only the head all-gather below still runs.
    full_heads = heads
    if tp_axis is not None:
        tp = int(list(mesh.shape)[tp_axis])
        if heads_presharded:
            full_heads = heads * tp  # q/k/v arrive already head-sharded; reassemble to this many
        else:
            assert heads % tp == 0, f"heads={heads} must split evenly over tp={tp}"
            q = ttnn.mesh_partition(q, dim=4, cluster_axis=tp_axis)
            k = ttnn.mesh_partition(k, dim=4, cluster_axis=tp_axis)
            v = ttnn.mesh_partition(v, dim=4, cluster_axis=tp_axis)
            heads = heads // tp

    width = heads * head_dim
    seq_local = w_local * t_full * h_full  # W-outer flatten of this chip's band
    # No tile-alignment requirement on seq_local: to_layout tile-pads the sequence dim transparently
    # and the gather/op operate on the logical shape, so a non-multiple-of-32 shard is exact (verified
    # against the host reference). The deterministic stages rely on this -- their (W/sp)*T*H is not
    # tile-aligned at any useful sp, unlike the stage-5 grid.
    if scale is None:
        scale = head_dim**-0.5
    kt, kh, kw = (min(kk, d) for kk, d in zip(kernel_size, dims))

    # DIFFVAE_BLOCK=1 (block-permute v1.1): 3-D block-permuted Q on TOP of the cheap t_inner K/V path.
    # to_seq already emits the W-outer (op) order, so a q_chunk block tiles (w_local, H, T) with op dims
    # (bw, bh, bt); the K/V prep and the output un-flatten below are reused UNCHANGED (the win was lost in
    # v1 by reordering K/V to plain-strided -- an 18x costlier gather). The compact block box is what fixes
    # the fused-sdpa super-linearity. The per-device global-W origin rides the offset tensor, applied to the
    # op OUTER (T) axis in the kernel (windowed_loop_geometry.hpp). RoPE commutes with the permute.
    # DIFFVAE_GNA=1 additionally sets the GNA stride to the Q block, which is the setting that collapses
    # each chunk's box to a single shared window (perfectly block-sparse). The block picker then optimizes
    # for that regime instead. This changes the ATTENTION, not just the schedule: queries inside a block
    # share one window rather than each being centered, so expect a quality delta -- measure it.
    _gna = os.environ.get("DIFFVAE_GNA") == "1"
    op_block = None
    if os.environ.get("DIFFVAE_BLOCK") == "1" and os.environ.get("DIFFVAE_SP_FUSED", "0") == "1":
        logger.info(f"""DIFFVAE_BLOCK={os.environ.get("DIFFVAE_BLOCK")}""")
        logger.info(f"""DIFFVAE_SP_FUSED={os.environ.get("DIFFVAE_SP_FUSED", "0")}""")

        _blk = _pick_block(t_full, h_full, w_local, gna=_gna)
        if _blk is not None:
            op_block = (_blk[2], _blk[1], _blk[0])  # op-order (w_local, H, T) block dims = (bw, bh, bt)
        logger.info(
            f"NA3D block config: _blk={_blk}, op_block={op_block}, gna={_gna}, "
            f"t_full={t_full}, h_full={h_full}, w_local={w_local}"
        )

    # Flatten W-outer so a contiguous sequence is a W-band; heads merged then re-split so the spatial
    # reorder is one 5D permute. ``w_`` is this chip's W extent (K/V and Q are the same shard here).
    #
    # DIFFVAE_SP_TINNER=1 makes T (the smallest axis — band frames) the INNERMOST flatten axis instead
    # of H. The fused kernel's conservative box blows up along whichever axis a q_chunk spans, so a
    # 128-query chunk over H (272) is a 128-tall strip (box_h = kh+127) while over T (~4-8) it wraps
    # into a compact (H-span x T) patch (box ~ kw x (kh+H_span-1) x t_full) -- ~7x fewer box keys at
    # 1080p. W stays outermost so the SP all-gather still rebuilds the full W-outer sequence. This is
    # a pure host reorder: the op decodes coords from the grid arg, agnostic to which axis is which.
    t_inner = os.environ.get("DIFFVAE_SP_TINNER", "1") == "1"

    # GNA stride in OP-axis order, matching how neighborhood_3d and op_block are permuted below. An
    # explicit gna_stride (or DIFFVAE_GNA_STRIDE="st,sh,sw") is PHYSICAL (t,h,w) and permutes the same way
    # the kernel does; DIFFVAE_GNA=1 instead takes the stride straight from the block, already op-order.
    _stride_env = os.environ.get("DIFFVAE_GNA_STRIDE")

    # (1,1,1) is NOT a choice -- it is the shipped architecture, i.e. no stride at all. Normalizing
    # it to None here is what lets a caller pass its resolved stride unconditionally: DIFFVAE_GNA
    # and DIFFVAE_GNA_STRIDE still apply underneath a trivial one, exactly as they do when the
    # argument is omitted. Without this, a caller that always passes a value silently turns the
    # DIFFVAE_GNA block stride off, and nothing fails to say so. TODO: REMOVE THIS HACK!
    if gna_stride == (1, 1, 1):
        gna_stride = None
    if gna_stride is None and _stride_env:
        gna_stride = tuple(int(v) for v in _stride_env.split(","))
    if gna_stride is not None:
        _st, _sh, _sw = gna_stride
        op_stride = (_sw, _sh, _st) if t_inner else (_sw, _st, _sh)
    elif _gna and op_block is not None:
        op_stride = op_block
    else:
        op_stride = None

    def _rm_wouter(x: ttnn.Tensor, w_: int) -> ttnn.Tensor:
        """(B, NH, S, HD) T-outer, any layout -> ROW_MAJOR (NH, w_, ., ., HD) W-outer.

        Extracted so the tiled and the paged K/V paths cannot drift on the axis order -- getting
        that order wrong does not fail, it silently attends to the wrong neighbourhood.
        """
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (heads, t_full, h_full, w_, head_dim))
        return ttnn.permute(x, (0, 3, 2, 1, 4) if t_inner else (0, 3, 1, 2, 4))

    def to_seq_flat(x: ttnn.Tensor, w_: int) -> ttnn.Tensor:
        """(B, NH, S, HD) T-outer -> the same, W-outer. One permute; both reshapes are views."""
        x = _rm_wouter(x, w_)
        x = ttnn.reshape(x, (batch, heads, w_ * t_full * h_full, head_dim))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _page_split(page_axis: int) -> int:
        """How many inner-axis elements go in one gathered page.

        The collective's cost turned out to depend on this and not, as assumed, to fall as the page
        grows: one page per row (9984 B here) gathers 39% slower than the tiled path's 2048 B pages.
        Everything after the gather is a free ROW_MAJOR view, so the page is tunable -- the only hard
        constraint is that a shard still contributes whole W rows, which holds for any divisor of the
        inner axis. DIFFVAE_KV_RM_PAGE is a target in BYTES; the largest divisor that fits wins, and
        0 (the default) means one page per row.
        """
        target = int(os.environ.get("DIFFVAE_KV_RM_PAGE", "0"))
        if target <= 0:
            return page_axis
        row_bytes = head_dim * 2  # bf16; the RM path forbids bfloat8_b (it is TILE-only)
        fits = [d for d in range(1, page_axis + 1) if page_axis % d == 0 and d * row_bytes <= target]
        return max(fits) if fits else 1

    def to_seq_paged(x: ttnn.Tensor, w_: int) -> ttnn.Tensor:
        """W-outer and ROW_MAJOR, already in the paging the fused reader gathers from.

        Same buffer order to_seq_flat produces; the only difference is where the row boundary is
        drawn, so this is the flat form's reshape with the innermost axis folded into the page.
        """
        x = _rm_wouter(x, w_)
        rows, page_axis = (h_full, t_full) if t_inner else (t_full, h_full)
        k = _page_split(page_axis)
        # (w, rows, page_axis, hd) -> rows carry the (page_axis / k) split, so the concat still cuts
        # on a W boundary and the post-gather merge back to one page per row is a view.
        return ttnn.reshape(x, (batch, heads, w_ * rows * (page_axis // k), k * head_dim))

    def to_seq(x: ttnn.Tensor, w_: int) -> ttnn.Tensor:
        if flat_seq:
            return to_seq_flat(x, w_)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch, t_full, h_full, w_, width))
        # (B, w_, H, T, width) [T innermost] or (B, w_, T, H, width) [H innermost].
        x = ttnn.permute(x, (0, 3, 2, 1, 4) if t_inner else (0, 3, 1, 2, 4))
        x = ttnn.reshape(x, (batch, w_ * t_full * h_full, heads, head_dim))
        x = ttnn.permute(x, (0, 2, 1, 3))  # (B, NH, S, HD)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    # K/V: gather this chip's W-band into the full W the window needs (chip order = W order, so the
    # concatenation rebuilds the full W-outer sequence). Q stays this chip's W-shard.
    #
    # DIFFVAE_CCL_PERSISTENT=1 routes the all-gathers through cached persistent output buffers + the
    # pre-created ping-pong semaphores (instead of a fresh buffer per call). Required for ttnn tracing:
    # a trace bakes absolute addresses, so a freshly-allocated gather output is clobbered on replay.
    persist_ccl = os.environ.get("DIFFVAE_CCL_PERSISTENT", "0") == "1"
    # DIFFVAE_KV_BF8=1 casts K/V to bfloat8_b before the gather. Both things it touches are on the
    # critical path: the gather's page size is the tile size of the gathered dtype (1088 B against
    # bf16's 2048 B), and the attention then reads the smaller K/V. Cast BEFORE the collective --
    # after it the bytes have already moved and only the read benefits.
    kv_dtype = ttnn.bfloat8_b if os.environ.get("DIFFVAE_KV_BF8", "0") == "1" else None

    # DIFFVAE_PAD_GATHER=1: pad this shard's sequence to a whole number of tiles so the gather's
    # alignment test passes and it takes the single-dispatch path -- which is also the path that
    # forwards CCLManager's pre-created semaphores, and therefore the only one a trace can replay
    # (composite's all_broadcast allocates its own semaphores per call). The pad rows land interleaved
    # between shards; they are stripped in wrow below, in ROW_MAJOR, where the fused reader was going
    # to untilize anyway -- so the strip costs a strided copy rather than a retile.
    # DIFFVAE_AG_HYPERPARAMS=1 hands the K/V and head gathers the tuned chunking (chunks_per_sync,
    # workers and buffers per channel) instead of the op defaults. The fabric caps num_links at 2 on
    # the size-8 axis -- only 2 eth channels reach across it -- so chunking is the remaining knob on
    # collectives that measure ~3.4 GB/s here.
    ag_hyper = os.environ.get("DIFFVAE_AG_HYPERPARAMS", "0") == "1"
    fused_reader = os.environ.get("DIFFVAE_SP_FUSED", "0") == "1"
    # DIFFVAE_KV_RM_GATHER=1: gather K/V ALREADY ROW_MAJOR and already paged, instead of tilizing the
    # shard, gathering tiled, and then untilizing the (8x larger) gathered result. Measured on the
    # 1080p stage-5 grid, that untilize -- kv-wrow -- is 125.8 ms per block, 5.21 GB of read-reorder-
    # write, and it buys nothing the gather could not have delivered directly.
    #
    # The concat still rebuilds the full W. A shard's rows are (w_local, rows)-ordered with w
    # outermost, so device-order concatenation on dim 2 lands global row (s*w_local + w)*rows + i --
    # exactly the W-outer paging wrow produced. Pages get BIGGER, not smaller: the page is the
    # innermost axis fused with head_dim (9984 B at this grid) against a tile's 2048 B, which is also
    # why the pad-to-a-whole-tile dance the tiled gather needs disappears with it.
    rm_gather = os.environ.get("DIFFVAE_KV_RM_GATHER", "0") == "1" and fused_reader and flat_seq
    if rm_gather and kv_dtype is not None:
        msg = "DIFFVAE_KV_RM_GATHER and DIFFVAE_KV_BF8 are mutually exclusive: bfloat8_b exists only in TILE layout"
        raise ValueError(msg)
    pad_gather = os.environ.get("DIFFVAE_PAD_GATHER") == "1" and fused_reader and not rm_gather and seq_local % 32 != 0
    seq_pad = ((seq_local + 31) // 32) * 32 if pad_gather else seq_local

    def gathered(x: ttnn.Tensor) -> ttnn.Tensor:
        if rm_gather:
            # No typecast and no pad: neither applies to a ROW_MAJOR gather, and wrow below is skipped
            # because this already IS wrow's output.
            out = ccl_manager.all_gather(
                to_seq_paged(x, w_local),
                dim=2,
                mesh_axis=sp_axis,
                use_hyperparams=ag_hyper,
                use_persistent_buffer=persist_ccl,
            )
            rows, page_axis = (h_full, t_full) if t_inner else (t_full, h_full)
            # Merge the page split back so the reader sees one page per row. Same buffer, new strides.
            return ttnn.reshape(out, (batch, heads, w_full * rows, page_axis * head_dim))
        seq = to_seq(x, w_local)
        if kv_dtype is not None and seq.get_dtype() != kv_dtype:
            cast = ttnn.typecast(seq, kv_dtype)
            ttnn.deallocate(seq)
            seq = cast
        if pad_gather:
            padded = ttnn.pad(seq, [(0, 0), (0, 0), (0, seq_pad - seq_local), (0, 0)], value=0.0)
            ttnn.deallocate(seq)
            seq = padded
        return ccl_manager.all_gather(
            seq, dim=2, mesh_axis=sp_axis, use_hyperparams=ag_hyper, use_persistent_buffer=persist_ccl
        )

    with _sp_w_prof(mesh, "kv-allgather", category=decode_tree.ALLGATHER):
        tk = gathered(k)
        tv = gathered(v)
    with _deep_prof(mesh, "q-to-seq", category=decode_tree.RESHAPE):
        tq = to_seq(q, w_local)
    if op_block is not None:
        from .block_permute import to_block_order_tt

        with _deep_prof(mesh, "q-block-permute", category=decode_tree.RESHAPE):
            tq = to_block_order_tt(tq, (w_local, h_full, t_full), op_block)  # W-outer op order -> block order

    # Block mode: the offset tensor carries the per-device global-W origin (shard*w_local) for the box;
    # strided mode: the per-device global token position (shard*seq_local).
    off_tt = ccl_manager.get_shard_offsets(sp, w_local if op_block is not None else seq_local, sp_axis)

    # DIFFVAE_SP_FUSED=1 runs the fast fused (neighborhood_gather) kernel instead of the streamed op:
    # K/V become the W-row-paged ROW_MAJOR layout the fused reader gathers from (grid is W-outer, so a
    # page is an inner-axis (h_full) row and the flattened (T,H) = w_full*t_full). Q + the per-device
    # offset are unchanged (verified: fused honours windowed_q_token_offset).
    use_fused = fused_reader
    prog_config = None
    if use_fused:

        def wrow(x: ttnn.Tensor) -> ttnn.Tensor:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            if pad_gather:
                # Drop each shard's pad rows. Both reshapes are stride changes in ROW_MAJOR; only the
                # slice copies.
                x = ttnn.reshape(x, (heads, sp, seq_pad, head_dim))
                x = ttnn.slice(x, [0, 0, 0, 0], [heads, sp, seq_local, head_dim])
                x = ttnn.reshape(x, (batch, heads, sp * seq_local, head_dim))
            # Page = the innermost-axis row: (w,h)-paged T-rows when t_inner, else (w,t)-paged H-rows.
            if t_inner:
                return ttnn.reshape(x, (batch, heads, w_full * h_full, t_full * head_dim))
            return ttnn.reshape(x, (batch, heads, w_full * t_full, h_full * head_dim))

        if not rm_gather:
            with _deep_prof(mesh, "kv-wrow (retile gathered K/V)", category=decode_tree.RESHAPE):
                tk, tv = wrow(tk), wrow(tv)
        grid_dev = mesh.compute_with_storage_grid_size()
        # Larger q_chunk = fewer chunks = the per-chunk fixed overhead (mask-gen + reader/compute setup)
        # amortizes over more queries. The box grows with q_chunk (its k-tiles are streamed, so L1 is
        # fine), so this trades a slightly larger box for far fewer chunks. Measured on the 1080p stage-5
        # grid: 32 -> 5559ms is the min at 128 (64: 6342, 256: 5994 regresses); PCC unchanged (0.9992).
        # Default 128; DIFFVAE_SDPA_QCHUNK/KCHUNK override for other grids.
        prog_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid_dev.x, grid_dev.y),
            exp_approx_mode=False,
            q_chunk_size=(
                op_block[0] * op_block[1] * op_block[2]
                if op_block is not None
                else int(os.environ.get("DIFFVAE_SDPA_QCHUNK", 128))
            ),
            k_chunk_size=int(os.environ.get("DIFFVAE_SDPA_KCHUNK", 32)),
        )

    with _sp_w_prof(mesh, "fused-sdpa" if use_fused else "op-sdpa", category=decode_tree.SDPA):
        attended = ttnn.transformer.scaled_dot_product_attention(
            tq,
            tk,
            tv,
            is_causal=False,
            # Grid axis order must match the flatten (innermost last): (w,h,t) when t_inner else (w,t,h).
            neighborhood_3d=((w_full, h_full, t_full, kw, kh, kt) if t_inner else (w_full, t_full, h_full, kw, kt, kh)),
            neighborhood_gather=use_fused,
            # Block-permuted Q: op-order block dims + the per-device W origin on the offset tensor (above).
            # No neighborhood_w_shard: the block path clamps each op axis with nb_T / nb_W directly.
            neighborhood_block=op_block,
            # GNA: op-order stride. None => stride 1 on every axis, i.e. standard neighborhood attention.
            neighborhood_stride=op_stride,
            scale=scale,
            windowed_q_token_offset=0,
            windowed_q_token_offset_tensor=off_tt,
            program_config=prog_config,
        )
    for tensor in (tq, tk, tv):
        ttnn.deallocate(tensor)

    # TP-over-heads: reassemble the full head-width from the tp shards while still (B, NH, S, HD)
    # rank-4 (all_gather supports rank 4). Device order along tp_axis is head order, so gathering on
    # the head axis rebuilds [head0 | head1 | ...]; downstream sees the full width as without TP.
    if tp_axis is not None:
        with _sp_w_prof(mesh, "head-allgather", category=decode_tree.ALLGATHER):
            attended = ccl_manager.all_gather(
                attended, dim=1, mesh_axis=tp_axis, use_hyperparams=ag_hyper, use_persistent_buffer=persist_ccl
            )
        heads = full_heads
        width = heads * head_dim

    # Block mode: un-permute block order back to the W-outer (op) sequence, then the same un-flatten runs.
    if op_block is not None:
        from .block_permute import from_block_order_tt

        with _deep_prof(mesh, "unblock-permute", category=decode_tree.RESHAPE):
            attended = from_block_order_tt(attended, (w_local, h_full, t_full), op_block)

    if flat_seq:
        # Straight to (tokens, NH*HD): one permute puts T,H,W back in order and heads next to
        # head_dim, so the caller's out-projection reads the result as a view.
        with _deep_prof(mesh, "attn-unflatten", category=decode_tree.RESHAPE):
            attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
            if t_inner:
                attended = ttnn.reshape(attended, (heads, w_local, h_full, t_full, head_dim))
                attended = ttnn.permute(attended, (3, 2, 1, 0, 4))  # (T, H, W_local, NH, HD)
            else:
                attended = ttnn.reshape(attended, (heads, w_local, t_full, h_full, head_dim))
                attended = ttnn.permute(attended, (2, 3, 1, 0, 4))  # (T, H, W_local, NH, HD)
            flat = ttnn.reshape(attended, (t_full * h_full * w_local, heads * head_dim))
            tiled = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
        return tiled

    # (B, NH, seq_local, HD) -> W-outer volume -> (B, T, H, W_local, width), sharded. The un-flatten
    # must mirror to_seq's axis order: (w,h,t) when t_inner, else (w,t,h).
    attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
    attended = ttnn.permute(attended, (0, 2, 1, 3))  # (B, seq_local, NH, HD)
    if t_inner:
        attended = ttnn.reshape(attended, (batch, w_local, h_full, t_full, width))
        return ttnn.permute(attended, (0, 3, 2, 1, 4))  # (B, T, H, W_local, width)
    attended = ttnn.reshape(attended, (batch, w_local, t_full, h_full, width))
    return ttnn.permute(attended, (0, 2, 3, 1, 4))  # (B, T, H, W_local, width)


def neighborhood_attention_3d_op_sp_sharded(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    dims: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    sp_axis: int,
    ccl_manager,
    scale: float | None = None,
) -> ttnn.Tensor:
    """SP-over-T with SHARDED input AND output, for full-stage sequence parallelism.

    Unlike :func:`neighborhood_attention_3d_op_sp` (replicated in/out, used to split just one
    attention call), this keeps the sequence sharded across the whole stage: ``q``/``k``/``v`` are
    THIS chip's contiguous T-slice ``(B, T/sp, H, W, NH, HD)`` and the return is this chip's T-slice
    of the output ``(B, T/sp, H, W, NH*HD)``. K/V are all-gathered to the full grid internally (the T
    window reaches a few frames past a shard edge), but Q and the output stay sharded so the block's
    residual adds and the surrounding pointwise ops (norm, proj, SwiGLU) all stay 1/sp in both
    compute and memory. ``dims`` is the FULL ``(T, H, W)``.

    Each chip is told its global frame origin via a per-device offset tensor, and its Q shard is
    already RoPE'd for its global positions by the caller (the cos/sin tables are sharded the same
    way). Requires whole-frame shards (T divisible by the mesh axis) and a tile-aligned shard origin.
    """
    mesh = q.device()
    sp = int(list(mesh.shape)[sp_axis])
    t_full, h_full, w_full = dims
    batch, t_local, h, w, heads, head_dim = tuple(q.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"
    assert (h, w) == (h_full, w_full), f"q spatial dims {(h, w)} != dims {(h_full, w_full)}"
    assert t_local * sp == t_full, f"T={t_full} must split evenly over sp={sp} (got T_local={t_local})"
    hw = h_full * w_full
    seq_full = t_full * hw
    seq_local = t_local * hw
    width = heads * head_dim
    tile_height = 32
    assert seq_local % tile_height == 0, f"shard origin (T/sp)*H*W={seq_local} must be a multiple of {tile_height}"
    if scale is None:
        scale = head_dim**-0.5
    kernels = tuple(min(kk, d) for kk, d in zip(kernel_size, dims))

    def to_seq(x: ttnn.Tensor, s_: int) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch, s_, heads, head_dim))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    # K/V: gather this chip's frame-shard into the full grid the window needs.
    tk = ccl_manager.all_gather(to_seq(k, seq_local), dim=2, mesh_axis=sp_axis, use_hyperparams=False)
    tv = ccl_manager.all_gather(to_seq(v, seq_local), dim=2, mesh_axis=sp_axis, use_hyperparams=False)
    tq = to_seq(q, seq_local)  # Q stays sharded

    off_tt = ccl_manager.get_shard_offsets(sp, seq_local, sp_axis)

    attended = ttnn.transformer.scaled_dot_product_attention(
        tq,
        tk,
        tv,
        is_causal=False,
        neighborhood_3d=(t_full, h_full, w_full, *kernels),
        scale=scale,
        windowed_q_token_offset=0,
        windowed_q_token_offset_tensor=off_tt,
    )
    for tensor in (tq, tk, tv):
        ttnn.deallocate(tensor)

    # (B, NH, seq_local, HD) -> (B, T_local, H, W, width), still sharded over T on this chip.
    attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
    attended = ttnn.permute(attended, (0, 2, 1, 3))
    return ttnn.reshape(attended, (batch, t_local, h_full, w_full, width))


def neighborhood_attention_3d(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    scale: float | None = None,
    device_plan: NA3DDevicePlan | None = None,
    chunk_budget: int = DEFAULT_CHUNK_BUDGET,
    ccl_manager=None,
    backend: str = "gather",
    sp_axis: int | None = None,
    gna_stride: tuple[int, int, int] | None = None,
) -> ttnn.Tensor:
    """3D neighborhood attention on device.

    ``q``/``k``/``v`` are ``(B, T, H, W, num_heads, head_dim)``, already RMS-normed and
    RoPE'd. Pass ``scale=1.0`` when the caller has pre-scaled Q, as the DiffVAE blocks do.
    Returns ``(B, T, H, W, num_heads * head_dim)`` in ROW_MAJOR layout.

    Either input layout is accepted. Callers building q/k/v with matmuls arrive in TILE,
    callers coming from a gather arrive in ROW_MAJOR, and the gathers below need ROW_MAJOR;
    normalizing here keeps that off every caller.

    ``chunk_budget`` caps the elements gathered per attention call, bounding peak memory
    independently of grid size. It changes no arithmetic: a group's tiles are independent, so
    splitting the batch is exact.

    When ``device_plan`` is sharded (see :class:`NA3DShard`) each chip evaluates a slice of every
    group and the results are gathered back here, so the return value is the same full volume on
    every chip either way. ``ccl_manager`` is only consulted when this builds its own plan; a
    plan passed in carries the manager it was built with.

    ``backend`` selects the executor. ``"gather"`` (default) is the grouped gather + dense masked
    attention above. ``"op"`` routes to :func:`neighborhood_attention_3d_op`, which synthesizes the
    neighborhood mask inside the SDPA op and needs no gather, mask upload, or CCL. ``"op_sp"`` routes
    to :func:`neighborhood_attention_3d_op_sp`, the same op path with the attention split over T
    across ``sp_axis`` (needs ``ccl_manager`` and ``sp_axis``). The gather-only arguments
    (``device_plan``, ``chunk_budget``) do not apply to the op backends.

    ``gna_stride`` is the GNA query-group stride in PHYSICAL (t, h, w) sites; the trivial (1,1,1)
    means the shipped architecture, so callers may pass it to any backend. Only ``"bricked"``
    honours a real stride here -- the others have no stride parameter, so one aimed at them is
    REFUSED rather than dropped: silently ignoring it is how a caller ends up measuring standard NA
    and reporting it as GNA.
    """
    if backend == "bricked":
        # Our op: tokens in bricked site order, one tile row per 3D brick.
        from .neighborhood_attention import neighborhood_attention_3d_bricked

        return neighborhood_attention_3d_bricked(q, k, v, kernel_size=kernel_size, scale=scale, stride=gna_stride)
    assert gna_stride in (None, (1, 1, 1)), (
        f"backend {backend!r} has no stride parameter, so gna_stride={gna_stride} would be ignored; "
        f'use backend="bricked" (or the sharded executors, which take it directly)'
    )
    if backend == "op":
        return neighborhood_attention_3d_op(q, k, v, kernel_size=kernel_size, scale=scale)
    if backend == "fused":
        return neighborhood_attention_3d_op_fused(q, k, v, kernel_size=kernel_size, scale=scale)
    if backend == "op_sp":
        assert ccl_manager is not None and sp_axis is not None, "op_sp needs ccl_manager and sp_axis"
        return neighborhood_attention_3d_op_sp(
            q, k, v, kernel_size=kernel_size, sp_axis=sp_axis, ccl_manager=ccl_manager, scale=scale
        )
    if backend != "gather":
        raise ValueError(f"unknown NA3D backend {backend!r}; expected 'gather', 'op', 'op_sp', 'fused' or 'bricked'")

    batch, t, h, w, heads, head_dim = tuple(q.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"
    # The gathers below are ttnn.embedding, which validates that its table is bfloat16. Caught
    # here so the constraint reads as a property of this executor rather than surfacing as a
    # TT_FATAL from inside the op. Lifting it means replacing the gather, not casting: a quiet
    # downcast would make an fp32 caller think it had fp32 attention.
    assert (
        q.dtype == ttnn.bfloat16
    ), f"NA3D gathers rows with ttnn.embedding, which requires a bfloat16 table; got {q.dtype}"
    if device_plan is None:
        device_plan = cached_device_plan(
            (t, h, w), kernel_size, mesh_device=q.device(), dtype=q.dtype, ccl_manager=ccl_manager
        )
    assert (
        device_plan.shard is None or device_plan.ccl_manager is not None
    ), "a sharded plan needs the CCL manager it was built with to reassemble each group"
    if scale is None:
        scale = head_dim**-0.5
    if scale != 1.0:
        # Elementwise multiply wants TILE, so scale before the ROW_MAJOR conversion below.
        q = ttnn.multiply(ttnn.to_layout(q, ttnn.TILE_LAYOUT), scale)

    # Fold heads into the row width so each of q/k/v is a (T*H*W, heads*head_dim) table that
    # ttnn.embedding can gather rows out of. The fold merges the last two dims, which is a
    # pure stride change in ROW_MAJOR but would need re-tiling in TILE.
    width = heads * head_dim
    tables = [ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (t * h * w, width)) for x in (q, k, v)]

    outputs = []
    for group in device_plan.groups:
        # A group's tiles are independent, so they can run in slices of the batch. Run whole, a
        # group's gathered K and V scale with the entire grid: 9 GB at 1920x1088 against 31.8 GB
        # of DRAM, which cannot coexist with the rest of a block. Chunking bounds peak memory by
        # a budget rather than by resolution. Grids small enough come out as one chunk, so the
        # shapes under test exercise this same path.
        n_tiles = group.local_tiles
        per_tile = group.n_keys * width  # elements gathered per tile, for each of K and V
        tiles_per_chunk = max(1, min(n_tiles, chunk_budget // max(1, 2 * per_tile)))

        chunks = []
        for start in range(0, n_tiles, tiles_per_chunk):
            tiles = min(tiles_per_chunk, n_tiles - start)
            # A chunk slices the leading dim, which is contiguous. When the chunk is the whole
            # group the slice is skipped: it would copy the plan's index tensor for nothing.
            if tiles == n_tiles:
                chunk_indices = (group.query_indices, group.key_indices)
            else:
                chunk_indices = tuple(
                    ttnn.slice(rows, [start, 0], [start + tiles, count])
                    for rows, count in (
                        (group.query_indices, group.local_queries),
                        (group.key_indices, group.n_keys),
                    )
                )

            gathered = []
            for table, index, count in (
                (tables[0], chunk_indices[0], group.local_queries),
                (tables[1], chunk_indices[1], group.n_keys),
                (tables[2], chunk_indices[1], group.n_keys),
            ):
                # (tiles, count) index -> (tiles, count, width), then split width into heads.
                # Splitting the innermost dim is a pure stride change in ROW_MAJOR.
                rows = ttnn.embedding(index, table, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=q.dtype)
                rows = ttnn.reshape(rows, (tiles, count, heads, head_dim))
                # (tiles, seq, heads, dim) -> (tiles, heads, seq, dim): SDPA wants heads ahead of seq.
                rows = ttnn.permute(rows, (0, 2, 1, 3))
                gathered.append(ttnn.to_layout(rows, ttnn.TILE_LAYOUT))
            # Never deallocated: on the single-chunk path these *are* the cached plan's tensors,
            # and freeing them would break every later block sharing that geometry.
            del chunk_indices

            # The mask is [1, 1, Nq, Nk] and broadcasts over the tile batch, so a chunk uses the
            # group's mask unchanged however many tiles it holds.
            attended = ttnn.transformer.scaled_dot_product_attention(
                gathered[0], gathered[1], gathered[2], attn_mask=group.mask, is_causal=False, scale=1.0
            )
            for tensor in gathered:
                ttnn.deallocate(tensor)

            attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
            attended = ttnn.permute(attended, (0, 2, 1, 3))
            chunks.append(ttnn.reshape(attended, (tiles, group.local_queries, width)))

        # Chunks are joined here rather than gathered individually: the chunking is a local memory
        # decision and must not reach the fabric, or the reassembled order would depend on it.
        local = chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=0)
        for tensor in chunks:
            if tensor is not local:
                ttnn.deallocate(tensor)
        outputs.append(ttnn.reshape(local, (group.local_tiles * group.local_queries, width)))

    # One stack per chip, then one gather per mesh axis for the whole attention call. Gathering
    # per group instead needs two CCL programs per group, and compiling ~100 of those costs more
    # than the attention they parallelize.
    local_stack = ttnn.concat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
    for tensor in outputs:
        if tensor is not local_stack:
            ttnn.deallocate(tensor)
    stacked = _gather_stack(local_stack, device_plan)

    restored = ttnn.embedding(device_plan.restore_indices, stacked, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=q.dtype)
    ttnn.deallocate(stacked)
    return ttnn.reshape(restored, (batch, t, h, w, width))
