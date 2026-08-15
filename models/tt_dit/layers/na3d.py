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

import math
from dataclasses import dataclass

import torch

import ttnn

# Cap on one tile group's [Nq, Nk] score block. Bounds both the additive mask allocation and
# the score materialization; the tile search shrinks axes until the product fits.
DEFAULT_SCORE_BUDGET = 2**22


def window_bounds(length: int, kernel: int) -> tuple[list[int], list[int]]:
    """Per-index ``(start, end)`` of the attended window along one axis.

    Implements NATTEN's constant-size inward-shifted window: the start is the query index
    less half the kernel, clamped so the window never leaves ``[0, length)``. When the axis
    is shorter than the kernel every query attends to the whole axis.
    """
    kernel = min(kernel, length)
    half = kernel // 2
    last_start = length - kernel
    starts = [min(max(i - half, 0), last_start) for i in range(length)]
    return starts, [s + kernel for s in starts]


def _tile_lengths(
    dims: tuple[int, int, int],
    kernels: tuple[int, int, int],
    budget: int,
    caps: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    """Per-axis query-tile lengths keeping one tile's score block under ``budget``.

    Halves whichever axis is largest relative to its kernel, since that is the axis whose
    span is cheapest to shrink: a tile of length ``t`` spans ``t + k - 1`` keys, so the waste
    factor is ``(t + k - 1) / k`` and shrinking a long axis with a small kernel helps most.

    ``caps`` bounds each axis's key span; it is the buffer extent rather than the query
    extent when the volume carries a halo. Tile size is a budget heuristic, so a loose cap
    costs accuracy in the estimate, never correctness.
    """
    tiles = list(dims)
    spans = caps if caps is not None else dims

    def score_block(candidate: list[int]) -> int:
        n_q = math.prod(candidate)
        n_k = math.prod(min(d, t + k - 1) for t, k, d in zip(candidate, kernels, spans))
        return n_q * n_k

    while score_block(tiles) > budget and max(tiles) > 1:
        axis = max(range(3), key=lambda a: tiles[a] / kernels[a])
        if tiles[axis] <= 1:
            break
        tiles[axis] = max(1, (tiles[axis] + 1) // 2)
    return tiles[0], tiles[1], tiles[2]


AxisGeometry = tuple[tuple[int, ...], tuple[int, ...], int]
"""Per-axis window bounds of one query tile, relative to that tile's key span, and the span."""


@dataclass(frozen=True)
class TileGroup:
    """Query tiles that share one window geometry, and therefore one additive mask.

    Grouping matters: with the inward-shift rule an axis has only three regimes (leading
    clamp, interior slide, trailing clamp), so a full volume collapses to at most 27 distinct
    masks in 3D no matter how many tiles there are.
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
    query_dims: tuple[int, int, int] | None = None
    query_origin: tuple[int, int, int] = (0, 0, 0)

    @property
    def output_dims(self) -> tuple[int, int, int]:
        """Extent this plan produces. Narrower than ``dims`` when the volume carries a halo."""
        return self.query_dims if self.query_dims is not None else self.dims

    @property
    def waste_factor(self) -> float:
        """Dense score elements evaluated per element an exact NA kernel would evaluate.

        1.0 is a perfect kernel; the dense formulation always pays more because a tile's key
        span covers the union of its queries' windows. Useful for choosing tile sizes.

        Measured against the queries this plan answers, not its buffer, so a sharded plan's
        factor is comparable to the whole volume's.
        """
        out = self.output_dims
        ideal = math.prod(out) * math.prod(min(k, d) for k, d in zip(self.kernels, self.dims))
        dense = sum(len(g.query_slices) * g.n_queries * g.n_keys for g in self.groups)
        return dense / ideal


@dataclass(frozen=True)
class AxisShard:
    """One axis of a volume split across devices: which queries this device answers.

    ``start``/``stop`` are indices into the *global* axis. The halo this device needs is not
    a free parameter — it follows from the window rule (:func:`required_halo`).
    """

    length: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        assert 0 <= self.start < self.stop <= self.length, f"bad shard [{self.start},{self.stop}) of {self.length}"


def required_halo(shard: AxisShard, kernel: int) -> tuple[int, int]:
    """Neighbour rows this shard must hold on each side to answer its own queries.

    Derived from the global window bounds, so it collapses to zero at a true volume edge and
    is ``kernel // 2`` at an interior seam. A caller sizing a halo exchange asks here rather
    than assuming ``kernel // 2`` everywhere, which would over-request at the volume border.
    """
    kernel = min(kernel, shard.length)
    starts, ends = window_bounds(shard.length, kernel)
    return shard.start - starts[shard.start], ends[shard.stop - 1] - shard.stop


def _axis_entries(
    starts: list[int],
    ends: list[int],
    *,
    q_begin: int,
    q_end: int,
    step: int,
    origin: int,
    buffer_length: int,
    span: int | None = None,
) -> list[tuple[slice, slice, AxisGeometry]]:
    """Tile ``[q_begin, q_end)`` of one axis, emitting slices in buffer coordinates.

    ``starts``/``ends`` are always the *global* window bounds. Recomputing them on a shard's
    own extent is the one mistake that matters here: it would treat every interior seam as a
    volume edge, where the window clamps inward instead of sliding, and be wrong at each one
    without failing.
    """
    entries = []
    for begin in range(q_begin, q_end, step):
        stop = min(begin + step, q_end)
        span_start, span_stop = starts[begin], ends[stop - 1]
        assert (
            span_start >= origin and span_stop <= origin + buffer_length
        ), f"tile [{begin},{stop}) needs keys [{span_start},{span_stop}) outside buffer [{origin},{origin + buffer_length})"

        if span is not None:
            # Every tile takes the same number of keys so a mesh can run one program: slide the
            # span inward from the buffer edge rather than shrink it, and let the mask drop the
            # surplus. At a volume edge the surplus is halo pad, which is why the exchange's
            # padding_mode never reaches the result.
            assert span >= span_stop - span_start, f"span {span} cannot cover [{span_start},{span_stop})"
            span_start = max(origin, min(span_start, origin + buffer_length - span))
            assert (
                span_start <= starts[begin] and span_start + span >= span_stop
            ), f"uniform span {span} does not fit tile [{begin},{stop}) in buffer of {buffer_length}"
            span_stop = span_start + span

        geometry = (
            tuple(s - span_start for s in starts[begin:stop]),
            tuple(e - span_start for e in ends[begin:stop]),
            span_stop - span_start,
        )
        entries.append((slice(begin - origin, stop - origin), slice(span_start - origin, span_stop - origin), geometry))
    return entries


def _assemble_groups(per_axis: list[list[tuple[slice, slice, AxisGeometry]]]) -> tuple[TileGroup, ...]:
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
    return tuple(groups)


def plan_na3d(
    dims: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    *,
    budget: int = DEFAULT_SCORE_BUDGET,
) -> NA3DPlan:
    """Group the volume's query tiles by window geometry."""
    kernels = tuple(min(k, d) for k, d in zip(kernel_size, dims))
    bounds = [window_bounds(d, k) for d, k in zip(dims, kernels)]
    tile = _tile_lengths(dims, kernels, budget)

    per_axis = [
        _axis_entries(*bounds[axis], q_begin=0, q_end=dims[axis], step=tile[axis], origin=0, buffer_length=dims[axis])
        for axis in range(3)
    ]
    return NA3DPlan(dims=dims, kernels=kernels, tile=tile, groups=_assemble_groups(per_axis))


def uniform_halo(length: int, parts: int, kernel: int) -> int:
    """One halo width that satisfies every shard of an even split.

    ``neighbor_pad_async`` pads all devices identically and fills from ``padding_mode`` where
    no neighbour exists, so a mesh program needs a single width rather than the per-shard
    minimum. Taking the max keeps every device's buffer the same shape; the fill an edge
    device receives is never read, because window bounds are global and stop at the volume.
    """
    kernel = min(kernel, length)
    edges = [round(i * length / parts) for i in range(parts + 1)]
    shards = [AxisShard(length=length, start=a, stop=b) for a, b in zip(edges, edges[1:])]
    return max(max(required_halo(s, kernel)) for s in shards)


def plan_na3d_sharded(
    shards: tuple[AxisShard, AxisShard, AxisShard],
    kernel_size: tuple[int, int, int],
    *,
    halo: tuple[int, int, int] | None = None,
    uniform_spans: bool = False,
    budget: int = DEFAULT_SCORE_BUDGET,
) -> NA3DPlan:
    """Plan one device's share of a volume split over ``shards``.

    The returned plan addresses the *local buffer* — this device's queries plus the halo rows
    a neighbour exchange delivers — so the executors consume it unchanged. Window bounds come
    from the global axis, which is what makes an interior seam behave as interior.

    ``halo`` is the width actually present on each side of the buffer, per axis; it defaults
    to the per-shard minimum. Pass :func:`uniform_halo` when a mesh program pads every device
    alike — the surplus rows at a volume edge are pad values, and a global-bounds plan never
    indexes them.

    ``uniform_spans`` fixes every tile's key count so that all devices in a mesh produce
    identically shaped work. Without it an edge shard's windows clamp and its spans come out
    shorter than an interior shard's, which is correct but undispatchable as one SPMD program.

    ``dims`` is the buffer, ``output_dims`` the queries answered, and ``query_origin`` where
    the queries sit inside the buffer.
    """
    lengths = tuple(s.length for s in shards)
    kernels = tuple(min(k, d) for k, d in zip(kernel_size, lengths))
    needed = tuple(required_halo(s, k) for s, k in zip(shards, kernels))
    if halo is None:
        halos = needed
    else:
        halos = tuple((h, h) for h in halo)
        for axis, ((have_l, have_r), (need_l, need_r)) in enumerate(zip(halos, needed)):
            assert (
                have_l >= need_l and have_r >= need_r
            ), f"axis {axis} halo {(have_l, have_r)} is below the window rule's {(need_l, need_r)}"
    origins = tuple(s.start - hl for s, (hl, _) in zip(shards, halos))
    buffers = tuple((s.stop + hr) - (s.start - hl) for s, (hl, hr) in zip(shards, halos))
    queries = tuple(s.stop - s.start for s in shards)

    bounds = [window_bounds(d, k) for d, k in zip(lengths, kernels)]
    tile = _tile_lengths(queries, kernels, budget, caps=buffers)
    # A tile's natural span is ``tile + kernel - 1``, shrinking only where the window clamps at
    # a volume edge. Holding it fixed costs masked-out keys and buys one shape for every device.
    spans = tuple(min(t + k - 1, b) for t, k, b in zip(tile, kernels, buffers)) if uniform_spans else (None,) * 3

    per_axis = [
        _axis_entries(
            *bounds[axis],
            q_begin=shards[axis].start,
            q_end=shards[axis].stop,
            step=tile[axis],
            origin=origins[axis],
            buffer_length=buffers[axis],
            span=spans[axis],
        )
        for axis in range(3)
    ]
    return NA3DPlan(
        dims=buffers,
        kernels=kernels,
        tile=tile,
        groups=_assemble_groups(per_axis),
        query_dims=queries,
        query_origin=tuple(hl for hl, _ in halos),
    )


def group_mask(group: TileGroup, *, dtype: torch.dtype, device: torch.device | str = "cpu") -> torch.Tensor:
    """Additive ``[1, 1, Nq, Nk]`` mask for one group: 0 where visible, -inf where not."""
    visible_per_axis = []
    for starts, ends, span in group.geometry:
        start = torch.tensor(starts, device=device)
        end = torch.tensor(ends, device=device)
        # The span, not ``end.max()``: a uniform-span tile carries surplus keys past its last
        # window, and they must be masked rather than silently dropped from the mask's width.
        key_index = torch.arange(span, device=device)
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
) -> torch.Tensor:
    """Host executor for a plan. ``q``/``k``/``v`` are ``(B, T, H, W, NH, HD)``.

    Pass ``scale=1.0`` when the caller has already scaled Q, which is what the DiffVAE
    blocks do — applying it again here would square the factor.
    """
    batch, t, h, w, heads, head_dim = q.shape
    if scale is None:
        scale = head_dim**-0.5
    if scale != 1.0:
        q = q * scale
    if plan is None:
        plan = plan_na3d((t, h, w), kernel_size)

    # A sharded plan is handed the halo'd buffer but answers only its own queries, so the
    # output is the narrower extent and slices are rebased off the halo.
    origin = plan.query_origin
    out = v.new_empty((batch, *plan.output_dims, heads, head_dim))
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
            written = tuple(slice(s.start - o, s.stop - o) for s, o in zip(q_slice, origin))
            out[:, written[0], written[1], written[2]] = attended.view(batch, heads, *tile_shape, head_dim).permute(
                0, 2, 3, 4, 1, 5
            )
    return out


def _flat_indices(dims: tuple[int, int, int], block: tuple[slice, slice, slice]) -> torch.Tensor:
    """Row indices into a ``T*H*W``-row table for one (t, h, w) block, in row-major order."""
    t, h, w = dims
    grid = torch.arange(t * h * w).reshape(t, h, w)
    return grid[block[0], block[1], block[2]].reshape(-1)


@dataclass
class NA3DDevicePlan:
    """Uploaded index tensors and masks for one ``(dims, kernel)`` on one mesh.

    Built once and cached: the index arithmetic is shape-dependent but weight-independent, so
    every block sharing a grid shape and kernel reuses this.
    """

    plan: NA3DPlan
    query_indices: tuple[ttnn.Tensor, ...]
    key_indices: tuple[ttnn.Tensor, ...]
    masks: tuple[ttnn.Tensor, ...]
    tiles_per_group: tuple[int, ...]
    restore_indices: ttnn.Tensor


def build_device_plan(
    plan: NA3DPlan,
    *,
    mesh_device,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> NA3DDevicePlan:
    """Upload a plan's gather indices and additive masks.

    Queries are gathered rather than sliced because a real grid has tens of thousands of
    tiles: as slices that is one op each, as a gather it is one op per *group*. The tiles of
    a group partition into the batch dimension, so a group becomes a single batched attention
    call sharing one mask.
    """
    query_indices, key_indices, masks, tiles_per_group = [], [], [], []
    plan_order = []

    for group in plan.groups:
        q_rows = torch.cat([_flat_indices(plan.dims, block) for block in group.query_slices])
        k_rows = torch.cat([_flat_indices(plan.dims, block) for block in group.key_slices])
        plan_order.append(q_rows)
        tiles_per_group.append(len(group.query_slices))

        query_indices.append(
            ttnn.from_torch(
                q_rows.to(torch.int32).reshape(1, -1),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        )
        key_indices.append(
            ttnn.from_torch(
                k_rows.to(torch.int32).reshape(1, -1),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        )
        # bfloat16's most-negative value stands in for -inf: exp() of it underflows to zero,
        # which is what the masked softmax needs, and it survives the dtype round trip.
        mask = group_mask(group, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
        masks.append(ttnn.from_torch(mask, device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT))

    # Query tiles partition the volume, so concatenating groups visits every voxel exactly
    # once — just in plan order. argsort maps that back to volume order in one final gather.
    order = torch.cat(plan_order)
    expected = math.prod(plan.output_dims)
    assert order.numel() == expected, f"plan covers {order.numel()} of {expected} voxels"
    # Sorting by buffer-flat index is row-major order over the query box, halo'd or not, so
    # the same argsort restores a sharded plan's output.
    restore = torch.argsort(order)

    return NA3DDevicePlan(
        plan=plan,
        query_indices=tuple(query_indices),
        key_indices=tuple(key_indices),
        masks=tuple(masks),
        tiles_per_group=tuple(tiles_per_group),
        restore_indices=ttnn.from_torch(
            restore.to(torch.int32).reshape(1, -1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
    )


def build_mesh_device_plan(
    plans: list[NA3DPlan],
    *,
    mesh_device,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> NA3DDevicePlan:
    """One device plan whose gather indices and masks differ per device.

    ``plans`` is one plan per device in row-major mesh order. Every device runs the same
    program, so the plans must agree on group count and on each group's query/key counts —
    what they may differ in is the index *values* and the mask, which is exactly what a shard's
    position in the volume changes. :func:`plan_na3d_sharded` with ``uniform_spans=True`` is
    what makes that agreement hold; a mismatch here means the geometry needs more than one
    program and cannot run as a single mesh dispatch.
    """
    rows, cols = tuple(mesh_device.shape)
    assert len(plans) == rows * cols, f"{len(plans)} plans for a {rows}x{cols} mesh"

    def signature(plan: NA3DPlan):
        return (plan.dims, plan.output_dims, tuple((g.n_queries, g.n_keys, len(g.query_slices)) for g in plan.groups))

    distinct = {signature(p) for p in plans}
    assert len(distinct) == 1, f"{len(distinct)} distinct plan shapes; a mesh dispatch needs one"

    def shard(per_device: list[torch.Tensor], tt_dtype, layout):
        hosts = [ttnn.from_torch(t, dtype=tt_dtype, layout=layout) for t in per_device]
        return ttnn.to_device(ttnn.from_host_shards(hosts, ttnn.MeshShape(rows, cols)), mesh_device)

    reference = plans[0]
    query_indices, key_indices, masks = [], [], []
    for index in range(len(reference.groups)):
        rows_q, rows_k, group_masks = [], [], []
        for plan in plans:
            group = plan.groups[index]
            rows_q.append(
                torch.cat([_flat_indices(plan.dims, block) for block in group.query_slices])
                .to(torch.int32)
                .reshape(1, -1)
            )
            rows_k.append(
                torch.cat([_flat_indices(plan.dims, block) for block in group.key_slices])
                .to(torch.int32)
                .reshape(1, -1)
            )
            group_masks.append(group_mask(group, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32))
        query_indices.append(shard(rows_q, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))
        key_indices.append(shard(rows_k, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))
        masks.append(shard(group_masks, dtype, ttnn.TILE_LAYOUT))

    restores = []
    for plan in plans:
        order = torch.cat([torch.cat([_flat_indices(plan.dims, b) for b in g.query_slices]) for g in plan.groups])
        assert order.numel() == math.prod(plan.output_dims)
        restores.append(torch.argsort(order).to(torch.int32).reshape(1, -1))

    return NA3DDevicePlan(
        plan=reference,
        query_indices=tuple(query_indices),
        key_indices=tuple(key_indices),
        masks=tuple(masks),
        tiles_per_group=tuple(len(g.query_slices) for g in reference.groups),
        restore_indices=shard(restores, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    )


_PLAN_CACHE: dict[tuple, NA3DDevicePlan] = {}


def cached_device_plan(
    dims: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    *,
    mesh_device,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> NA3DDevicePlan:
    """Device plan for one geometry, built once per process.

    A plan depends only on the grid, the kernel and the dtype — never on weights — but it
    uploads index tables and masks, so rebuilding it per block (every block of a stack shares
    a geometry) would dominate a decode. Keyed on the device id as well, since the uploaded
    tensors belong to one mesh.
    """
    key = (tuple(dims), tuple(kernel_size), dtype, id(mesh_device))
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = build_device_plan(plan_na3d(tuple(dims), tuple(kernel_size)), mesh_device=mesh_device, dtype=dtype)
        _PLAN_CACHE[key] = plan
    return plan


def neighborhood_attention_3d(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    scale: float | None = None,
    device_plan: NA3DDevicePlan | None = None,
) -> ttnn.Tensor:
    """3D neighborhood attention on device.

    ``q``/``k``/``v`` are ``(B, T, H, W, num_heads, head_dim)``, already RMS-normed and
    RoPE'd. Pass ``scale=1.0`` when the caller has pre-scaled Q, as the DiffVAE blocks do.
    Returns ``(B, T, H, W, num_heads * head_dim)`` in ROW_MAJOR layout.

    Either input layout is accepted. Callers building q/k/v with matmuls arrive in TILE,
    callers coming from a gather arrive in ROW_MAJOR, and the gathers below need ROW_MAJOR;
    normalizing here keeps that off every caller.
    """
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
        device_plan = cached_device_plan((t, h, w), kernel_size, mesh_device=q.device(), dtype=q.dtype)
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
    for group, q_index, k_index, mask, n_tiles in zip(
        device_plan.plan.groups,
        device_plan.query_indices,
        device_plan.key_indices,
        device_plan.masks,
        device_plan.tiles_per_group,
    ):
        gathered = []
        for table, index, count in (
            (tables[0], q_index, group.n_queries),
            (tables[1], k_index, group.n_keys),
            (tables[2], k_index, group.n_keys),
        ):
            rows = ttnn.embedding(index, table, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=q.dtype)
            rows = ttnn.reshape(rows, (n_tiles, count, heads, head_dim))
            # (tiles, seq, heads, dim) -> (tiles, heads, seq, dim): SDPA wants heads ahead of seq.
            rows = ttnn.permute(rows, (0, 2, 1, 3))
            gathered.append(ttnn.to_layout(rows, ttnn.TILE_LAYOUT))

        attended = ttnn.transformer.scaled_dot_product_attention(
            gathered[0], gathered[1], gathered[2], attn_mask=mask, is_causal=False, scale=1.0
        )
        for tensor in gathered:
            ttnn.deallocate(tensor)

        attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
        attended = ttnn.permute(attended, (0, 2, 1, 3))
        outputs.append(ttnn.reshape(attended, (n_tiles * group.n_queries, width)))

    stacked = ttnn.concat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
    for tensor in outputs:
        if tensor is not stacked:
            ttnn.deallocate(tensor)

    restored = ttnn.embedding(device_plan.restore_indices, stacked, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=q.dtype)
    ttnn.deallocate(stacked)
    # The output is the queries answered, which is narrower than the input volume whenever
    # the plan is a shard reading halo rows it does not write back.
    return ttnn.reshape(restored, (batch, *device_plan.plan.output_dims, width))
