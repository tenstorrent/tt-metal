# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""NA3D executor backed by the ``neighborhood_scaled_dot_product_attention`` device op.

Same contract as the other executors in ``na3d.py``: ``q``/``k``/``v`` are
``(B, T, H, W, num_heads, head_dim)``, already RMS-normed and RoPE'd with Q pre-scaled, and the
return is ``(B, T, H, W, num_heads * head_dim)`` in ROW_MAJOR. Selected with ``backend="bricked"``.

The difference is the site ORDER. The op consumes tokens bricked -- 32 consecutive sites are a
compact 3D box rather than a pencil along width -- so one tile row is one brick of video and a
query tile's context window is a handful of long reads instead of 121 short ones. See
``neighborhood_permute``.

Queries and keys span DIFFERENT regions on the W-sharded path, as they do in ``na3d.py``'s
sharded executor: a query needs a widened KEY region -- its window reaches past the shard seam --
but never a widened QUERY region, because the halo's own queries belong to the neighbour, which
computes them itself. So K and V are halo-exchanged and Q is not, and the op is told the
difference through ``query_extent``/``query_origin``. It addresses two brick grids: the resident
one for K, V and the gather, the query one for Q and the output.

Before that split Q was widened like K and V, and the op computed 76 resident columns to keep 60
-- 21% of every query discarded -- with a halo exchange on Q purely to satisfy a shape that was
then sliced away. Filling Q's halo locally instead of over the fabric was tried first and is
SLOWER (75.6 ms against the exchange's 34.2, decode 9213 -> 9565 ms): the halo columns are dead,
but ANY local fill copies the 60/76 the shard owns while the collective moves only the 16/76 that
crosses a seam. Not filling it at all is the only thing that helps.

This module converts in and out PER CALL unless ``already_bricked=True``. That flag is the
hoist: Q/K/V arrive in bricked site order from a conversion at stage entry, so this call only
halo-exchanges K/V on the ``W_br`` axis (whole bricks, no 7-D permute) and returns still-bricked.
Stage 5 converts back once at exit. The per-call spans remain so an un-hoisted run still names
the permute it is paying for.
"""

from __future__ import annotations

import os

import ttnn

from ..utils import decode_tree
from .neighborhood_permute import SITES_PER_BRICK, brick_count, brick_grid, to_bricked, to_natural

# Plans depend on no weights but do upload an index table, so rebuilding one per block would
# dominate. Keyed on the geometry, exactly as the op's program cache is.
_PLAN_CACHE: dict = {}


def _deep_prof(device, key: str, *, category: str | None = None):
    """The same span helper the reference executor uses, so both break down side by side in the
    decode tree. Inert unless DIFFVAE_BLOCK_PROF is set: each span costs two device syncs."""
    from ..models.vae.diffvae_ltx_stage5 import deep_prof

    return deep_prof(device, key, category=category)


def _window_origin(group_index, stride, window, volume, snap=0):
    """Host mirror of window_origin_on_axis. Only used to BUILD the interior mask, which the
    kernel then reads; the kernel keeps its own copy of the rule for boundary bricks."""
    first = group_index * stride
    last = min(first + stride - 1, volume - 1)
    centre = first + (last - first) // 2
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
    import torch

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
    import torch

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


def _cached_plan(volume, context_window, stride, brick, device, *, resident=None, shard_count=1, sp_axis=None):
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
    entry = _PLAN_CACHE.get(key)
    if entry is not None:
        return entry

    import torch

    from ..utils.tensor import from_torch

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
    _PLAN_CACHE[key] = first
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


def neighborhood_attention_3d_bricked(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int] | None = None,
    scale: float | None = None,
) -> ttnn.Tensor:
    batch, time_extent, height_extent, width_extent, head_count, head_dim = tuple(query.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"

    # PHYSICAL (t, h, w), NOT the op's axis order: it is passed to the op unpermuted below, while
    # na3d permutes its own gna_stride. (1,1,1) is the shipped architecture: every query centred on
    # its own window. Larger shares one window across each group, which is what removes the
    # per-query mask -- see the module docstring. Callers own the knob; this module reads no env.
    if stride is None:
        stride = (1, 1, 1)

    # DIFFVAE_NA_WINDOW overrides the architectural window. Enlarging it to a whole number of
    # bricks is what lets every gathered brick be classified wholly in or wholly out, so the
    # interior mask becomes memsets instead of per-site window evaluation. It changes what the
    # model sees, so it is an experiment knob, not a default.
    window_env = os.environ.get("DIFFVAE_NA_WINDOW")
    if window_env:
        kernel_size = tuple(int(part) for part in window_env.split(","))

    volume = (time_extent, height_extent, width_extent)
    # An axis shorter than the window is attended to in full, matching the op's own clamping.
    context_window = tuple(min(window, extent) for window, extent in zip(kernel_size, volume))
    # DIFFVAE_NA_BRICK overrides the derived brick. A brick 1 deep in time never needs time
    # padding, and output_frames = 8*latent_T - 7 is ALWAYS odd, so a 2-deep brick pads (and
    # therefore full-tensor copies q, k and v) on every single block.
    brick_env = os.environ.get("DIFFVAE_NA_BRICK")
    brick = (
        tuple(int(part) for part in brick_env.split(","))
        if brick_env
        else tuple(ttnn.transformer.neighborhood_choose_brick(context_window))
    )
    if scale is None:
        scale = head_dim**-0.5

    device = query.device()
    plan = _cached_plan(volume, context_window, stride, brick, device)
    bricked_sites = brick_count(volume, brick) * SITES_PER_BRICK
    channels = head_count * head_dim

    def to_op_layout(tensor: ttnn.Tensor) -> ttnn.Tensor:
        """(B,T,H,W,heads,head_dim) -> (B, 1, bricked_sites, heads*head_dim) in TILE.

        The op reads site-major, so heads never move: the 3D brick reorder is the only data
        movement here. Transposing heads against sites to satisfy a heads-major op cost 24.6 ms a
        block at stage-5 size -- a third of all layout time -- and bought no arithmetic.
        """
        rows = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        volume_form = ttnn.reshape(rows, (batch, time_extent, height_extent, width_extent, channels))
        bricked = to_bricked(volume_form, volume=volume, brick=brick)
        # Sites are the TILE ROW axis and heads are columns, so one tile row is one brick.
        site_major = ttnn.reshape(bricked, (batch, 1, bricked_sites, channels))
        return ttnn.to_layout(site_major, ttnn.TILE_LAYOUT)

    with _deep_prof(device, "brick-permute (q,k,v)", category=decode_tree.RESHAPE):
        query_op = to_op_layout(query)
        key_op = to_op_layout(key)
        value_op = to_op_layout(value)

    _tp_trace(
        device,
        f"about to call the op: chunk={plan['query_chunk_bricks']} "
        f"gather={plan['gather_brick_count']} kv_tiles={_tiles_per_kv_chunk(plan['gather_brick_count'])}",
    )
    with _deep_prof(device, "neighborhood-sdpa", category=decode_tree.SDPA):
        attended = ttnn.transformer.neighborhood_scaled_dot_product_attention(
            query_op,
            key_op,
            value_op,
            plan["gather_origin_tensor"],
            interior_mask=plan["interior_mask_tensor"],
            volume=volume,
            context_window=context_window,
            stride=stride,
            brick=brick,
            query_chunk_bricks=plan["query_chunk_bricks"],
            head_count=head_count,
            scale=scale,
            tiles_per_kv_chunk=_tiles_per_kv_chunk(plan["gather_brick_count"]),
        )

    with _deep_prof(device, "unbrick-permute", category=decode_tree.RESHAPE):
        # Site-major on the way out too, so this is a reshape rather than a transpose.
        rows = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
        merged = ttnn.reshape(rows, (batch, bricked_sites, channels))
        natural = to_natural(merged, volume=volume, brick=brick)

    return natural


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
    best, best_gather = default, None
    for brick_time in range(1, SITES_PER_BRICK + 1):
        for brick_height in range(1, SITES_PER_BRICK + 1):
            if SITES_PER_BRICK % (brick_time * brick_height):
                continue
            brick_width = SITES_PER_BRICK // (brick_time * brick_height)
            if brick_time * brick_height * brick_width != SITES_PER_BRICK:
                continue
            # An ODD brick width gives an odd halo, and the halo exchange then cannot fold its
            # sticks up to 256 bytes -- `neighbor_pad` hangs at 128. See _halo_exchange.
            if brick_width % 2:
                continue
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
                    )
                    for index in range(shard_count)
                ]
            except (ValueError, RuntimeError):
                continue  # a brick the planner refuses for this geometry
            if any(plan["gather_brick_count"] != plans[0]["gather_brick_count"] for plan in plans):
                continue  # one program cannot serve shards that gather differently
            gather = plans[0]["gather_brick_count"]
            # Tie-breaks, in order: fewest gathered bricks, then smallest halo, then the deepest
            # brick in TIME. A smaller halo is less to exchange, brick-permute and drop, and that
            # one is reasoned. Preferring depth in time is MEASURED and not explained -- shapes
            # that gather identically and carry the same halo do not run at the same speed, and
            # the deeper time extent won. It is a preference, not a rule: re-measure it if the
            # volume, window or shard count changes.
            score = (gather, halo, -brick_time)
            if best_gather is None or score < best_gather:
                best, best_gather = brick, score

    from loguru import logger

    logger.info(
        f"[neighborhood] brick {best} gathers {best_gather[0] if best_gather else '?'} bricks "
        f"(choose_brick would pick {default})"
    )
    _BRICK_CHOICE_CACHE[key] = best
    return best


def _tp_trace(device, message: str) -> None:
    """Synchronise and log, so a hang inside the TP block names the op it hung ON.

    The decode-tree spans only print at teardown, which a hang never reaches; the watcher
    segfaults on this 32-chip fabric. Gated on DIFFVAE_TP_TRACE because each call is a full
    mesh sync. Temporary bring-up scaffolding.
    """
    if os.environ.get("DIFFVAE_TP_TRACE") != "1":
        return
    from loguru import logger

    ttnn.synchronize_device(device)
    logger.info(f"[tp-trace] {message}")


# The widest stick ``neighbor_pad_async`` actually moves, in bytes. MEASURED, not documented by
# the op: at a 16 KB stick the first 4352 bytes of each halo column arrive and the rest is zeros
# and uninitialised DRAM, silently -- no error, no hang, right shape. 4 KB passes, 8 KB does not
# (models/tt_dit/tests/unit/test_halo_exchange_geometry.py::test_halo_exchange_moves_whole_sticks
# is the sweep). The end symptom off this is catastrophically low PCC ( ~%)
MAX_HALO_STICK_BYTES = 4096


def _halo_split(stick_elements: int, element_bytes: int) -> int:
    """How many sub-columns to cut one halo stick into so the exchange stays inside the bound.

    Splits without moving a byte, the inverse of the W-fold in ``widened``: a ``[.., w, s]``
    buffer is contiguous, so ``[.., w * parts, s / parts]`` is the SAME memory, and padding
    ``parts``-scaled columns pads exactly the same halo. Halving keeps ``parts`` a divisor.

    Returns 1 whenever the stick already fits, so a configuration that was inside the bound keeps
    the exact call it had. Production under TP4 is one such: 32 sites x 64 channels x 2 B is 4 KB
    on the nose, which is why only the non-TP paths -- every stage-5 gate -- ever saw this.
    """
    parts = 1
    while stick_elements // parts * element_bytes > MAX_HALO_STICK_BYTES and (stick_elements // parts) % 2 == 0:
        parts *= 2
    return parts


def _halo_exchange(ccl_manager, tensor, *, dims, pad_left, pad_right, axes, neighbor_sems, num_links):
    """neighbor_pad with the TOPOLOGY PINNED, independent of the manager's setting.

    ``neighbor_pad_async`` deadlocks on ``Topology.Ring``. Measured: the same call completes on
    Linear and hangs on Ring, and it is not the channel width (128 B vs 512 B stick), the link
    count (1 vs 2), or the persistent output buffer -- each was ruled out on its own. The reference
    executor never meets this because it gathers K/V with ``all_gather``; this op is the only
    caller of neighbor_pad, so Ring here had never run.

    Every OTHER collective in the decode is an all_gather, which is ring-safe and where ring is
    actually worth its -1710.9 ms (kv-allgather 2162.8 -> 840.1, head-allgather 409.3 -> 129.9).
    So ring stays on for all of them and only this one call drops to Linear.

    DIFFVAE_NA_HALO_TOPOLOGY=ring hands it back the manager's topology, to retest in one run once
    neighbor_pad is fixed.
    """
    stick_bytes = int(tensor.shape[-1]) * tensor.element_size()
    assert stick_bytes <= MAX_HALO_STICK_BYTES, (
        f"a {stick_bytes}-byte halo stick exceeds the {MAX_HALO_STICK_BYTES} B neighbor_pad moves "
        f"intact; it would return the right shape with the tail of every halo column unwritten. "
        f"Split the trailing dim into sub-columns of the same memory first -- see _halo_split"
    )
    topology = ccl_manager.topology if os.environ.get("DIFFVAE_NA_HALO_TOPOLOGY") == "ring" else ttnn.Topology.Linear
    barrier = ccl_manager.get_barrier_semaphore(axes[0])
    buffer = ccl_manager.get_np_ping_pong_buffer(
        tensor.shape, dims, pad_left, pad_right, dtype=tensor.get_dtype(), t_front_pad=0
    )
    return ttnn.experimental.neighbor_pad_async(
        tensor,
        dims,
        pad_left,
        pad_right,
        "zeros",
        axes,
        neighbor_sems,
        [barrier],
        num_links=num_links,
        topology=topology,
        persistent_output_buffer=buffer,
        logical_h=0,
        t_front_pad=0,
    )


def neighborhood_attention_3d_bricked_w_sharded(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    *,
    dims: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    sp_axis: int,
    ccl_manager,
    scale: float | None = None,
    tp_axis: int | None = None,
    heads_presharded: bool = False,
    already_bricked: bool = False,
    brick: tuple[int, int, int] | None = None,
    stride: tuple[int, int, int] | None = None,
) -> ttnn.Tensor:
    """Spatial-W sharded NA3D. ``q``/``k``/``v`` are this chip's W-shard; ``dims`` is the FULL grid.

    Each chip widens its shard by a halo on both sides, runs the op over the widened region with
    window placement kept GLOBAL, and keeps only the columns it owns. Clamping therefore happens
    at the true volume boundary rather than at a shard seam -- the failure mode that would
    otherwise truncate the receptive field of every query within half a window of an internal
    edge and still return plausible video.

    The halo is symmetric because a mesh runs one program and every device must therefore hold
    the same resident extent. That puts the device at the low edge of the volume at a negative
    origin, which the op's signed shard origin handles.

    ``tp_axis`` adds TENSOR PARALLELISM OVER HEADS on a second, orthogonal mesh axis. Attention is
    independent per head, so each chip keeps ``heads/tp`` of them and they are all-gathered back
    right after the flash. Under the column-parallel qkv the projections already emit only this
    chip's heads (``heads_presharded``), and with one head left the caller hands us the flat
    ``(batch, heads, sites, head_dim)`` sequence rather than the 6-D volume -- both shapes are
    accepted below, because refusing the flat one refuses the whole fast path.

    ``already_bricked``: sites are already in bricked order (a caller-side hoist). Q/K/V are
    ``(batch, heads, bricked_sites, head_dim)``; the W halo is ``neighbor_pad`` on ``W_br``, and
    the return stays bricked. ``brick`` is then required so the caller and the op cannot disagree.

    ``stride`` is the GNA query-group stride in PHYSICAL (t, h, w) sites, defaulting to (1,1,1) --
    the shipped architecture, every query centred on its own window. It is the caller's knob: this
    module reads no environment for it, and a caller that also derives its own brick must pass the
    same stride here or the two brick choices can disagree.
    """
    shard_count = int(list(query.device().shape)[sp_axis])
    time_extent, height_extent = dims[0], dims[1]
    width_local = dims[2] // shard_count
    if already_bricked:
        batch, head_count, _, head_dim = tuple(query.shape)
        assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"
        assert brick is not None, "already_bricked needs the brick the stage converted with"
    else:
        if len(query.shape) == 4:
            # Flat (batch, heads, sites, head_dim). Sites run (t, h, w_local) exactly as the volume
            # form does, so this is a view, not a reorder.
            batch, head_count, _, head_dim = tuple(query.shape)
            volume_shape = (batch, time_extent, height_extent, width_local, head_count, head_dim)
            query, key, value = (ttnn.reshape(tensor, volume_shape) for tensor in (query, key, value))
        batch, time_extent, height_extent, width_local, head_count, head_dim = tuple(query.shape)
        assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"

    volume = dims
    assert (
        volume[2] == width_local * shard_count
    ), f"W {volume[2]} does not split into {shard_count} shards of {width_local}"

    context_window = tuple(min(window, extent) for window, extent in zip(kernel_size, volume))
    if stride is None:
        stride = (1, 1, 1)
    brick_env = os.environ.get("DIFFVAE_NA_BRICK")
    if brick is None:
        brick = (
            tuple(int(part) for part in brick_env.split(","))
            if brick_env
            else _choose_sharded_brick(volume, context_window, stride, width_local, shard_count)
        )
    if scale is None:
        scale = head_dim**-0.5

    halo = halo_sites(context_window[2], brick[2])
    assert halo <= width_local, (
        f"a {halo}-site halo exceeds the {width_local}-site shard: the window reaches past the "
        f"neighbour into its neighbour, which a single-hop exchange cannot serve"
    )
    resident = (time_extent, height_extent, width_local + 2 * halo)

    device = query.device()
    plan = _cached_plan(
        volume, context_window, stride, brick, device, resident=resident, shard_count=shard_count, sp_axis=sp_axis
    )
    channels = head_count * head_dim
    # K and V span the resident region (owned + halo); Q and the output span only what this shard
    # OWNS. Two brick counts because the op now addresses two grids -- see the module note.
    owned_volume = (time_extent, height_extent, width_local)
    bricked_sites = brick_count(resident, brick) * SITES_PER_BRICK
    query_bricked_sites = brick_count(owned_volume, brick) * SITES_PER_BRICK
    assert query_bricked_sites == plan["query_brick_count"] * SITES_PER_BRICK, (
        f"host bricks the owned region into {query_bricked_sites} sites but the plan says "
        f"{plan['query_brick_count'] * SITES_PER_BRICK}; the two must agree or Q is misaddressed"
    )
    if already_bricked:
        assert query.shape[-2] == query_bricked_sites, (
            f"already-bricked Q has {query.shape[-2]} sites but the owned brick grid is "
            f"{query_bricked_sites}; the stage converted with a different brick or T-pad"
        )
    # DIFFVAE_NA_HALO_LINKS overrides the link count for THIS halo exchange only, leaving every
    # other collective on the ccl_manager's setting. The halo hangs at channels=64 (one head per
    # chip under TP) where it runs fine at 256, and a two-link split of a narrow transfer is the
    # first thing to rule out -- one link per side with nothing to carry never signals its peer.
    num_links = int(os.environ.get("DIFFVAE_NA_HALO_LINKS", 0)) or max(1, ccl_manager.num_links)
    semaphore = ccl_manager.get_np_ping_pong_semaphore(sp_axis)

    def widened_bricked(tensor: ttnn.Tensor, lane: str = "?") -> ttnn.Tensor:
        """K/V halo in bricked order: ``neighbor_pad`` on ``W_br``, no 7-D permute.

        Last dim is ``32 * channels`` -- one brick of sites folded into the stick -- so no W-fold
        is needed to clear the 128 B width that hangs in natural order. Halo is 3 bricks rather
        than 6 sites at 1080p.

        But that stick has an UPPER bound too, and it is low: see ``_halo_split`` below.
        """
        t_br, h_br, w_br = brick_grid(owned_volume, brick)
        halo_br = halo // brick[2]
        parts = _halo_split(SITES_PER_BRICK * channels, tensor.element_size())
        _tp_trace(device, f"{lane}: untilize in (already_bricked, channels={channels}, parts={parts})")
        with _deep_prof(device, f"{lane}: untilize", category=decode_tree.RESHAPE):
            rows = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        with _deep_prof(device, f"{lane}: halo-exchange", category=decode_tree.ALLGATHER):
            grid5 = ttnn.reshape(rows, (batch, t_br, h_br, w_br * parts, SITES_PER_BRICK * channels // parts))
            exchanged = _halo_exchange(
                ccl_manager,
                grid5,
                dims=[3],
                pad_left=[halo_br * parts],
                pad_right=[halo_br * parts],
                axes=[sp_axis],
                neighbor_sems=[semaphore],
                num_links=[num_links],
            )
        _tp_trace(device, f"{lane}: neighbor_pad done -> {tuple(exchanged.shape)}")
        with _deep_prof(device, f"{lane}: tilize", category=decode_tree.RESHAPE):
            site_major = ttnn.reshape(exchanged, (batch, 1, bricked_sites, channels))
            out = ttnn.to_layout(site_major, ttnn.TILE_LAYOUT)
        _tp_trace(device, f"{lane}: tilized -> {tuple(out.shape)}")
        return out

    def widened(tensor: ttnn.Tensor, lane: str = "?") -> ttnn.Tensor:
        """This chip's shard plus a halo of each neighbour's edge, in op layout. K and V only:
        Q is bricked over the owned region alone and never comes through here.

        Three spans per lane, so the collective and the reorder can be read apart -- the whole
        point of the split is that they have different fixes. The untilize is its own span rather
        than folded into either one: both the halo exchange and ``to_bricked`` need ROW_MAJOR, so
        it is a prerequisite of both, and charging it to one would make that one look like the
        cost. Under DEEP these spans also serialize q, k and v against each other, so the parent's
        total inflates a little against a plain DIFFVAE_STAGE_TIMING run.
        """
        _tp_trace(device, f"{lane}: untilize in (channels={channels})")
        with _deep_prof(device, f"{lane}: untilize", category=decode_tree.RESHAPE):
            rows = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        # The exchange hangs at a 128-byte stick (channels=64, i.e. one head per chip under TP)
        # where it runs fine at 512 (channels=256, four heads). Neither the link count nor the
        # persistent buffer is behind it -- both were ruled out by measurement.
        #
        # So widen the stick without moving a byte: fold whole groups of W columns into the
        # channel axis. The buffer is [.., w, c] contiguous, so [.., w/f, c*f] is the SAME memory,
        # and padding f-column groups by halo/f pads exactly the same halo. Requires f to divide
        # both the halo and this chip's width, which 4 does at 1080p (halo 8, w_local 60).
        # 256 CHANNELS is the target (512 bytes at bf16) -- the width measured working. Halve the
        # candidate until it divides both the halo and this chip's width, so the fold is always
        # legal: 64 channels -> 4, and 4 divides halo 8 and w_local 60.
        fold = 1
        if channels < 256 and 256 % channels == 0:
            candidate = 256 // channels
            while candidate > 1 and (halo % candidate or width_local % candidate):
                candidate //= 2
            fold = candidate
        if os.environ.get("DIFFVAE_NA_HALO_FOLD") == "0":
            fold = 1
        exchange_channels, exchange_width, exchange_halo = channels * fold, width_local // fold, halo // fold
        _tp_trace(
            device,
            f"{lane}: about to neighbor_pad halo={exchange_halo} links={num_links} "
            f"fold={fold} stick={exchange_channels * 2}B",
        )
        # The fold reshape rides in the halo span: the fold exists only to widen this exchange's
        # stick, so its cost belongs to the exchange it serves.
        with _deep_prof(device, f"{lane}: halo-exchange", category=decode_tree.ALLGATHER):
            volume_form = ttnn.reshape(rows, (batch, time_extent, height_extent, exchange_width, exchange_channels))
            # DIFFVAE_NA_HALO_PERSISTENT=0 drops the persistent output buffer for this exchange.
            # The buffer is keyed on the tensor shape, and TP narrows the channels from 256 to
            # 64 -- a 128-byte stick -- so the pooled buffer is the next suspect after links.
            exchanged = _halo_exchange(
                ccl_manager,
                volume_form,
                dims=[3],
                pad_left=[exchange_halo],
                pad_right=[exchange_halo],
                axes=[sp_axis],
                neighbor_sems=[semaphore],
                num_links=[num_links],
            )
        _tp_trace(device, f"{lane}: neighbor_pad done -> {tuple(exchanged.shape)}")
        with _deep_prof(device, f"{lane}: brick-permute", category=decode_tree.RESHAPE):
            if fold > 1:  # unfold back to real columns; the same memory, read the other way
                exchanged = ttnn.reshape(
                    exchanged, (batch, time_extent, height_extent, width_local + 2 * halo, channels)
                )
            bricked = to_bricked(exchanged, volume=resident, brick=brick)
            _tp_trace(device, f"{lane}: to_bricked done -> {tuple(bricked.shape)}")
            site_major = ttnn.reshape(bricked, (batch, 1, bricked_sites, channels))
            out = ttnn.to_layout(site_major, ttnn.TILE_LAYOUT)
        _tp_trace(device, f"{lane}: tilized -> {tuple(out.shape)}")
        return out

    widen = widened_bricked if already_bricked else widened
    with _deep_prof(device, "halo+brick-permute (k,v)", category=decode_tree.RESHAPE):
        key_op = widen(key, "k")
        value_op = widen(value, "v")

    # Q is NOT widened: the halo's queries belong to the neighbour, which computes them itself,
    # and the op is told so via query_extent/query_origin below. So this is the brick permute
    # alone -- no exchange, and over 60 columns rather than 76. Already-bricked Q skips the
    # permute: it is a layout/reshape into the op's (B, 1, sites, C) TILE.
    with _deep_prof(device, "q-to-seq", category=decode_tree.RESHAPE):
        if already_bricked:
            rows = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
            site_major = ttnn.reshape(rows, (batch, 1, query_bricked_sites, channels))
            query_op = ttnn.to_layout(site_major, ttnn.TILE_LAYOUT)
            _tp_trace(device, f"q: already bricked -> {tuple(query_op.shape)}")
        else:
            rows = ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT)
            volume_form = ttnn.reshape(rows, (batch, time_extent, height_extent, width_local, channels))
            bricked = to_bricked(volume_form, volume=owned_volume, brick=brick)
            site_major = ttnn.reshape(bricked, (batch, 1, query_bricked_sites, channels))
            query_op = ttnn.to_layout(site_major, ttnn.TILE_LAYOUT)
            _tp_trace(device, f"q: bricked owned region -> {tuple(query_op.shape)}")

    with _deep_prof(device, "neighborhood-sdpa", category=decode_tree.SDPA):
        attended = ttnn.transformer.neighborhood_scaled_dot_product_attention(
            query_op,
            key_op,
            value_op,
            plan["gather_origin_tensor"],
            interior_mask=plan["interior_mask_tensor"],
            volume=volume,
            context_window=context_window,
            stride=stride,
            brick=brick,
            query_chunk_bricks=plan["query_chunk_bricks"],
            shard_extent=resident,
            # Representative only: the plan's SHAPES are uniform across shards, and each device
            # reads its own origin out of the sharded table above.
            shard_origin=(0, 0, -halo),
            # The queries this shard owns, as a sub-box of the resident region. Uniform across
            # the mesh -- every shard owns the same-shaped box at the same offset -- so unlike
            # shard_origin these are compile-time and one program still serves the whole mesh.
            query_extent=plan["query_extent"],
            query_origin=plan["query_origin"],
            head_count=head_count,
            scale=scale,
            tiles_per_kv_chunk=_tiles_per_kv_chunk(plan["gather_brick_count"]),
        )

    _tp_trace(device, f"op returned -> {tuple(attended.shape)}")
    with _deep_prof(device, "unbrick-permute", category=decode_tree.RESHAPE):
        rows = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
        merged = ttnn.reshape(rows, (batch, query_bricked_sites, channels))
        # Already the owned region: the op wrote only the queries this shard owns, so there is no
        # halo left to slice off -- that slice, and the queries behind it, are what this bought.
        # The hoisted path stays bricked; natural order is restored once at stage exit.
        owned = merged if already_bricked else to_natural(merged, volume=owned_volume, brick=brick)

    if tp_axis is not None:
        # Rebuild the full head width from the tp shards. Device order along tp_axis IS head
        # order, and heads are the channel axis here, so gathering the channels concatenates
        # [head0 | head1 | ...] -- the layout the replicated out-proj already expects.
        with _deep_prof(device, "head-allgather", category=decode_tree.ALLGATHER):
            sites_local = query_bricked_sites if already_bricked else time_extent * height_extent * width_local
            # One head per chip is what makes the next line a VIEW: the buffer is site-major with
            # the channels inside a site, so (b, heads, sites, head_dim) only coincides with it
            # when heads == 1. With more heads left per chip this would need a real transpose, and
            # reshaping instead would silently interleave them.
            assert head_count == 1, (
                f"TP over heads expects the column-parallel qkv (DIFFVAE_TP_PROJ) to leave one "
                f"head per chip; got {head_count}"
            )
            _tp_trace(device, "entering TP block")
            flat = ttnn.reshape(owned, (batch, head_count, sites_local, head_dim))
            _tp_trace(device, f"reshaped to {tuple(flat.shape)}")
            # Two things this block has to be careful about, both of which show up as a HANG
            # rather than an error:
            #
            # ttnn.reshape can hand back a VIEW over the same buffer, so freeing ``owned`` before
            # the gather can release the very DRAM the collective is about to read. Everything
            # else in this file frees through the ``is not`` guard for that reason; so does this.
            #
            # And the gather runs on TILES. ``owned`` arrives row-major straight out of
            # ttnn.slice, whereas the reference executor gathers a tiled rank-4 tensor at this
            # point. The caller retiles for the out-proj anyway, so tiling here is free.
            tiled = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
            _tp_trace(device, "tilized, about to all_gather dim=1")
            gathered = ccl_manager.all_gather(tiled, dim=1, mesh_axis=tp_axis, use_hyperparams=False)
            _tp_trace(device, f"all_gather done -> {tuple(gathered.shape)}")
            if tiled is not flat:
                ttnn.deallocate(tiled)
            if flat is not owned:
                ttnn.deallocate(flat)
            ttnn.deallocate(owned)

        # (b, heads, sites, head_dim) -> (b, 1, sites, heads * head_dim). The out-proj wants the
        # heads folded into the channels OF A SITE, which is a transpose of the head and site axes
        # rather than a reshape -- gathering on dim=1 is what buys the cheap collective, and this
        # is the move that pays for it. The reference executor does the same thing right after its
        # own head-allgather, under the name "attn-unflatten".
        with _deep_prof(device, "head-unflatten", category=decode_tree.RESHAPE):
            rows = ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT)
            _tp_trace(device, "untilized for permute")
            if rows is not gathered:
                ttnn.deallocate(gathered)
            moved = ttnn.permute(rows, (0, 2, 1, 3))
            _tp_trace(device, f"permuted -> {tuple(moved.shape)}")
            if moved is not rows:
                ttnn.deallocate(rows)
            heads_total = int(list(device.shape)[tp_axis]) * head_count
            owned = ttnn.reshape(moved, (batch, 1, sites_local, heads_total * head_dim))
            if moved is not owned:
                ttnn.deallocate(moved)
            # Tiled on the way out, because this branch returns the shape the caller was going to
            # reshape TO: _reshape_retiled short-circuits on an exact shape match and hands the
            # tensor straight to the out-proj, so a row-major return reaches minimal_matmul as-is
            # and trips "requires TILE layout". The no-TP path returns the 5-D volume, whose shape
            # never matches, which is why only this branch has to care.
            owned = ttnn.to_layout(owned, ttnn.TILE_LAYOUT)
            _tp_trace(device, f"retilized -> {tuple(owned.shape)}; TP block done")

    return owned
