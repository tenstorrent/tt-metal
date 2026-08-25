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

This module converts in and out PER CALL, which is the honest but slow arrangement: the permute
belongs once at stage entry, because everything between attentions is per-token and therefore
permutation-equivariant. Hoisting it is what turns the reference implementation's ~325 ms/block
of reshape+permute into ~0.6 ms. Getting the op running end to end came first; the hoist is
next, and the spans below are there to show what it would buy.
"""

from __future__ import annotations

import os

import ttnn

from ..utils import decode_tree
from .neighborhood_permute import SITES_PER_BRICK, brick_count, to_bricked, to_natural

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


def configured_stride() -> tuple[int, int, int]:
    """The GNA stride for stage 5, in PHYSICAL (time, height, width) sites.

    ``DIFFVAE_S5_GNA_STRIDE`` first, because ``DIFFVAE_GNA_STRIDE`` is read by na3d for EVERY
    stage, and the deterministic stages have smaller kernels: a stride legal for stage 5's 11^3
    window is rejected by a stage whose kernel is 7 ("neighborhood_stride t=8 must not exceed the
    effective kernel t=7"). That check reports OP-order axes, so a width stride surfaces as `t`,
    which makes the message doubly confusing. Setting the stage-5 knob leaves the other stages at
    whatever they were.

    (1,1,1) is the shipped architecture: every query centred on its own window.
    """
    for name in ("DIFFVAE_S5_GNA_STRIDE", "DIFFVAE_GNA_STRIDE"):
        value = os.environ.get(name)
        if value:
            return tuple(int(part) for part in value.split(","))
    return (1, 1, 1)


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
    return tuple(
        stride_extent // brick_extent if stride_extent % brick_extent == 0 else 1
        for stride_extent, brick_extent in zip(stride, brick)
    )


def _cached_plan(volume, context_window, stride, brick, device):
    query_chunk_bricks = _query_chunk_bricks(stride, brick)
    key = (volume, context_window, stride, brick, query_chunk_bricks, id(device))
    entry = _PLAN_CACHE.get(key)
    if entry is not None:
        return entry

    import torch

    plan = ttnn.transformer.neighborhood_plan(
        volume, context_window, stride, brick, query_chunk_bricks=query_chunk_bricks
    )
    plan["query_chunk_bricks"] = query_chunk_bricks
    origin_table = torch.tensor(plan["gather_origin_table"], dtype=torch.uint32).reshape(
        1, 1, plan["chunk_count"], plan["gather_origin_columns"]
    )
    plan["gather_origin_tensor"] = ttnn.from_torch(
        origin_table, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    from loguru import logger

    logger.info(
        f"[neighborhood] volume={volume} window={context_window} stride={stride} brick={brick} "
        f"chunk={query_chunk_bricks} bricks ({plan['bricks_per_query_chunk'] * SITES_PER_BRICK} queries) "
        f"bricks={plan['brick_count']} gather={plan['gather_brick_count']} tiles "
        f"({plan['gather_brick_count'] * SITES_PER_BRICK / (plan['bricks_per_query_chunk'] * SITES_PER_BRICK):.2f} "
        f"keys/query, waste "
        f"{plan['gather_brick_count'] * 32 / (context_window[0] * context_window[1] * context_window[2]):.2f}x)"
    )
    # Uploading these is the single largest win in the op. Generating masks on device instead
    # costs 43.5 ms of a 53.8 ms block at stage-5 size -- 81% of the whole attention -- because
    # every gathered brick that straddles the window edge is evaluated per site, per chunk, every
    # block. There are only 27 distinct patterns, they depend on nothing but geometry, and this
    # builds them once and keeps them resident.
    plan["interior_mask_tensor"] = ttnn.from_torch(
        _build_regime_masks(volume, context_window, stride, brick, query_chunk_bricks, plan),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    _PLAN_CACHE[key] = plan
    return plan


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

    # PHYSICAL (t, h, w), same convention as DIFFVAE_GNA_STRIDE elsewhere. (1,1,1) is the shipped
    # architecture: every query centred on its own window. Larger shares one window across each
    # group, which is what removes the per-query mask -- see the module docstring.
    if stride is None:
        stride = configured_stride()

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


_SHARDED_PLAN_CACHE: dict = {}


def _cached_sharded_plan(volume, context_window, stride, brick, resident, shard_count, sp_axis, device):
    """One plan per shard, with the per-device gather tables stacked for a sharded upload.

    Every device runs the SAME program, so the plan's shapes -- chunk count, gathered bricks --
    must agree across shards, and they do: the resident extent is uniform and every origin is
    brick-aligned, so only the origins themselves differ. Those ride the table, which is sharded
    over the mesh so each device reads its own.
    """
    query_chunk_bricks = _query_chunk_bricks(stride, brick)
    key = (volume, context_window, stride, brick, resident, shard_count, sp_axis, id(device))
    entry = _SHARDED_PLAN_CACHE.get(key)
    if entry is not None:
        return entry

    import torch

    from ..utils.tensor import from_torch

    owned_width = resident[2] - 2 * halo_sites(min(context_window[2], volume[2]), brick[2])
    plans = []
    for shard_index in range(shard_count):
        # The device at the low edge sits BELOW the volume by one halo. Those columns are real
        # storage holding nothing the volume contains; no query owns them, no window reaches them.
        origin_width = shard_index * owned_width - (resident[2] - owned_width) // 2
        plans.append(
            ttnn.transformer.neighborhood_plan(
                volume,
                context_window,
                stride,
                brick,
                query_chunk_bricks=query_chunk_bricks,
                shard_extent=resident,
                shard_origin=(0, 0, origin_width),
            )
        )

    first = plans[0]
    for shard_index, plan in enumerate(plans[1:], start=1):
        for field in ("chunk_count", "gather_brick_count", "gather_bricks", "volume_chunks"):
            assert plan[field] == first[field], (
                f"shard {shard_index} plans a different {field} than shard 0 "
                f"({plan[field]} vs {first[field]}); one program cannot serve both"
            )

    columns = first["gather_origin_columns"]
    stacked = torch.tensor([plan["gather_origin_table"] for plan in plans], dtype=torch.uint32).reshape(
        shard_count, 1, first["chunk_count"], columns
    )
    first["gather_origin_tensor"] = from_torch(
        stacked,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_axes=[sp_axis, None, None, None],
    )
    first["query_chunk_bricks"] = query_chunk_bricks

    from loguru import logger

    logger.info(
        f"[neighborhood] W-SHARDED x{shard_count}: volume={volume} resident={resident} "
        f"window={context_window} stride={stride} brick={brick} chunk={query_chunk_bricks} bricks "
        f"({first['bricks_per_query_chunk'] * SITES_PER_BRICK} queries) "
        f"gather={first['gather_brick_count']} tiles "
        f"({first['gather_brick_count'] / first['bricks_per_query_chunk']:.2f} keys/query)"
    )
    # Generated on device: the uploaded regime sets are enumerated against a single shard origin,
    # and here every shard has its own.
    first["interior_mask_tensor"] = None
    _SHARDED_PLAN_CACHE[key] = first
    return first


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
    """
    if len(query.shape) == 4:
        # Flat (batch, heads, sites, head_dim). Sites run (t, h, w_local) exactly as the volume
        # form does, so this is a view, not a reorder.
        batch, head_count, _, head_dim = tuple(query.shape)
        time_extent, height_extent = dims[0], dims[1]
        width_local = dims[2] // int(list(query.device().shape)[sp_axis])
        volume_shape = (batch, time_extent, height_extent, width_local, head_count, head_dim)
        query, key, value = (ttnn.reshape(tensor, volume_shape) for tensor in (query, key, value))
    batch, time_extent, height_extent, width_local, head_count, head_dim = tuple(query.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"

    shard_count = int(list(query.device().shape)[sp_axis])
    volume = dims
    assert (
        volume[2] == width_local * shard_count
    ), f"W {volume[2]} does not split into {shard_count} shards of {width_local}"

    context_window = tuple(min(window, extent) for window, extent in zip(kernel_size, volume))
    brick_env = os.environ.get("DIFFVAE_NA_BRICK")
    brick = (
        tuple(int(part) for part in brick_env.split(","))
        if brick_env
        else tuple(ttnn.transformer.neighborhood_choose_brick(context_window))
    )
    stride = configured_stride()
    if scale is None:
        scale = head_dim**-0.5

    halo = halo_sites(context_window[2], brick[2])
    assert halo <= width_local, (
        f"a {halo}-site halo exceeds the {width_local}-site shard: the window reaches past the "
        f"neighbour into its neighbour, which a single-hop exchange cannot serve"
    )
    resident = (time_extent, height_extent, width_local + 2 * halo)

    device = query.device()
    plan = _cached_sharded_plan(volume, context_window, stride, brick, resident, shard_count, sp_axis, device)
    channels = head_count * head_dim
    bricked_sites = brick_count(resident, brick) * SITES_PER_BRICK
    # DIFFVAE_NA_HALO_LINKS overrides the link count for THIS halo exchange only, leaving every
    # other collective on the ccl_manager's setting. The halo hangs at channels=64 (one head per
    # chip under TP) where it runs fine at 256, and a two-link split of a narrow transfer is the
    # first thing to rule out -- one link per side with nothing to carry never signals its peer.
    num_links = int(os.environ.get("DIFFVAE_NA_HALO_LINKS", 0)) or max(1, ccl_manager.num_links)
    semaphore = ccl_manager.get_np_ping_pong_semaphore(sp_axis)

    def widened(tensor: ttnn.Tensor, lane: str = "?") -> ttnn.Tensor:
        """This chip's shard plus a halo of each neighbour's edge, in op layout."""
        _tp_trace(device, f"{lane}: untilize in (channels={channels})")
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
        volume_form = ttnn.reshape(rows, (batch, time_extent, height_extent, exchange_width, exchange_channels))
        _tp_trace(
            device,
            f"{lane}: about to neighbor_pad halo={exchange_halo} links={num_links} "
            f"fold={fold} stick={exchange_channels * 2}B",
        )
        # DIFFVAE_NA_HALO_PERSISTENT=0 drops the persistent output buffer for this exchange. The
        # buffer is keyed on the tensor shape, and TP narrows the channels from 256 to 64 -- a
        # 128-byte stick -- so the pooled buffer is the next suspect after link count.
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
        if fold > 1:  # unfold back to real columns; the same memory, read the other way
            exchanged = ttnn.reshape(exchanged, (batch, time_extent, height_extent, width_local + 2 * halo, channels))
        bricked = to_bricked(exchanged, volume=resident, brick=brick)
        _tp_trace(device, f"{lane}: to_bricked done -> {tuple(bricked.shape)}")
        site_major = ttnn.reshape(bricked, (batch, 1, bricked_sites, channels))
        out = ttnn.to_layout(site_major, ttnn.TILE_LAYOUT)
        _tp_trace(device, f"{lane}: tilized -> {tuple(out.shape)}")
        return out

    with _deep_prof(device, "halo+brick-permute (q,k,v)", category=decode_tree.RESHAPE):
        query_op = widened(query, "q")
        key_op = widened(key, "k")
        value_op = widened(value, "v")

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
            head_count=head_count,
            scale=scale,
            tiles_per_kv_chunk=_tiles_per_kv_chunk(plan["gather_brick_count"]),
        )

    _tp_trace(device, f"op returned -> {tuple(attended.shape)}")
    with _deep_prof(device, "unbrick-permute", category=decode_tree.RESHAPE):
        rows = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
        merged = ttnn.reshape(rows, (batch, bricked_sites, channels))
        natural = to_natural(merged, volume=resident, brick=brick)
        _tp_trace(device, "unbricked, about to drop halo")
        # Drop the halo: those columns belong to a neighbour, which computed them itself.
        owned = ttnn.slice(
            natural,
            [0, 0, 0, halo, 0],
            [batch, time_extent, height_extent, halo + width_local, channels],
        )
        ttnn.deallocate(natural)

    if tp_axis is not None:
        # Rebuild the full head width from the tp shards. Device order along tp_axis IS head
        # order, and heads are the channel axis here, so gathering the channels concatenates
        # [head0 | head1 | ...] -- the layout the replicated out-proj already expects.
        with _deep_prof(device, "head-allgather", category=decode_tree.ALLGATHER):
            sites_local = time_extent * height_extent * width_local
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
