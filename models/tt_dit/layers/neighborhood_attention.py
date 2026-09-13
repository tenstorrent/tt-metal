# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The three device executors of 3D neighborhood attention in the LTX-2.5 DiffVAE decoder.

Same contract for all three: ``q``/``k``/``v`` are ``(B, T, H, W, num_heads, head_dim)``, already
RMS-normed and RoPE'd with Q pre-scaled, and the return is ``(B, T, H, W, num_heads * head_dim)`` in
ROW_MAJOR. Their plans -- everything that depends on the geometry and nothing on the weights --
live in ``neighborhood_attention_plan``.

* ``neighborhood_attention_3d_linear_order``: tokens in natural row-major order. Gathers each
  query group's key span with ``ttnn.embedding`` and runs dense masked SDPA over it, optionally
  splitting the query work across the mesh with K/V replicated. What stage 1 (index 0) runs, and
  the replicated oracle the sharded tests compare against.

* ``neighborhood_attention_3d_bricked``: the ``neighborhood_scaled_dot_product_attention`` device
  op over the whole volume on every chip. The op consumes tokens BRICKED -- 32 consecutive sites
  are a compact 3D box rather than a pencil along width -- so one tile row is one brick of video
  and a query tile's context window is a handful of long reads instead of 121 short ones. See
  ``neighborhood_permute``.

* ``neighborhood_attention_3d_bricked_w_sharded``: the same op over this chip's W-shard. Queries
  and keys span DIFFERENT regions: a query needs a widened KEY region -- its window reaches past
  the shard seam -- but never a widened QUERY region, because the halo's own queries belong to the
  neighbour, which computes them itself. So K and V are halo-exchanged and Q is not, and the op is
  told the difference through ``query_extent``/``query_origin``. It addresses two brick grids: the
  resident one for K, V and the gather, the query one for Q and the output.

  Widening Q too would make the op compute 76 resident columns to keep 60 -- 21% of every query
  discarded -- and filling Q's halo locally instead of over the fabric is SLOWER still (75.6 ms
  against the exchange's 34.2): the halo columns are dead, but ANY local fill copies the 60/76 the
  shard owns while the collective moves only the 16/76 that crosses a seam. Not filling it at all
  is the only thing that helps.

  It converts in and out PER CALL unless ``already_bricked=True``. That flag is the hoist: Q/K/V
  arrive in bricked site order from a conversion at stage entry, so this call only halo-exchanges
  K/V on the ``W_br`` axis (whole bricks, no 7-D permute) and returns still-bricked. Stage 5
  converts back once at exit. The per-call spans remain so an un-hoisted run still names the
  permute it is paying for.
"""

from __future__ import annotations

import os

import ttnn

from ..utils import decode_tree
from .neighborhood_attention_plan import (
    NA3DDevicePlan,
    _choose_sharded_brick,
    _tiles_per_kv_chunk,
    brick_override,
    cached_bricked_plan,
    cached_device_plan,
    halo_sites,
)
from .neighborhood_permute import SITES_PER_BRICK, brick_count, brick_grid, to_bricked, to_bricked_grid, to_natural

# Cap on the elements gathered for one attention call's K and V together. Peak device memory is
# what this bounds, and it is separate from the score budget above, which bounds a *tile's*
# window: a group can hold tens of thousands of tiles, and running them as one call is what
# scales with resolution. 2**29 elements is 1 GB in bfloat16 for K and V combined.
#
# The bound is per chip, so a sharded plan (see :class:`NA3DShard`) splits a group's tiles
# before this applies and needs proportionally fewer chunks for the same grid.
DEFAULT_CHUNK_BUDGET = 2**29


def _deep_prof(device, key: str, *, category: str | None = None):
    """The same span helper the reference executor uses, so both break down side by side in the
    decode tree. Inert unless DIFFVAE_BLOCK_PROF is set: each span costs two device syncs."""
    from ..models.vae.diffvae_ltx_stage5 import deep_prof

    return deep_prof(device, key, category=category)


def _compute_kernel_config() -> ttnn.WormholeComputeKernelConfig:
    """The numerics the fused SDPA runs with, so the bricked op is compared like for like.

    Until 2026-09-11 the op, left unspecified, fell back to ``DeviceComputeKernelConfig{}``: LoFi
    matmuls and the approximate exp. The general SDPA op that the replicated reference and the
    older block-permute executor run defaults to HiFi2 with an exact exp
    (``SDPAProgramConfig(exp_approx_mode=False)``). Measured 2026-09-10 on the stage-5 decode gate:
    the LoFi/approx default was a uniform ~5 % RMSE/sigma floor against the replicated reference
    (PCC 99.89 %, flat across frames and columns -- not a seam, not a shape). The op's binding now
    defaults to HiFi2/exact as well; this helper keeps the choice explicit and A/B-able.

    ``DIFFVAE_NA_FIDELITY=lofi|hifi2|hifi4`` and ``DIFFVAE_NA_APPROX_EXP=1`` are A/B knobs only.
    """
    fidelity = {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }[os.environ.get("DIFFVAE_NA_FIDELITY", "hifi2").lower()]
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=os.environ.get("DIFFVAE_NA_APPROX_EXP") == "1",
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


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


def neighborhood_attention_3d_linear_order(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    scale: float | None = None,
    device_plan: NA3DDevicePlan | None = None,
    chunk_budget: int = DEFAULT_CHUNK_BUDGET,
    ccl_manager=None,
    gna_stride: tuple[int, int, int] | None = None,
) -> ttnn.Tensor:
    """3D neighborhood attention with tokens in linear (natural) order: gather, then dense masked SDPA.

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

    ``gna_stride`` is the GNA query-group stride in PHYSICAL (t, h, w) sites; the trivial (1,1,1)
    means the shipped architecture, so callers may pass it. This executor has no stride parameter,
    so a real stride is REFUSED rather than dropped: silently ignoring it is how a caller ends up
    measuring standard NA and reporting it as GNA.
    """
    assert gna_stride in (None, (1, 1, 1)), (
        f"the linear-order executor has no stride parameter, so gna_stride={gna_stride} would be ignored; "
        f"use a bricked executor, which takes it directly"
    )

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

    # PHYSICAL (t, h, w), NOT the op's axis order: it is passed to the op unpermuted below.
    # (1,1,1) is the shipped architecture: every query centred on
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
    brick = brick_override(volume) or tuple(ttnn.transformer.neighborhood_choose_brick(context_window))
    if scale is None:
        scale = head_dim**-0.5

    device = query.device()
    plan = cached_bricked_plan(volume, context_window, stride, brick, device)
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
            compute_kernel_config=_compute_kernel_config(),
        )

    with _deep_prof(device, "unbrick-permute", category=decode_tree.RESHAPE):
        # Site-major on the way out too, so this is a reshape rather than a transpose.
        rows = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
        merged = ttnn.reshape(rows, (batch, bricked_sites, channels))
        natural = to_natural(merged, volume=volume, brick=brick)

    return natural


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

    Splits without moving a byte: a ``[.., w_br, s]`` brick grid is contiguous, so
    ``[.., w_br * parts, s / parts]`` is the SAME memory, and padding ``parts``-scaled columns
    pads exactly the same halo. Halving keeps ``parts`` a divisor.

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
    chip's heads (``heads_presharded``), and the caller may hand us the flat HEAD-major
    ``(batch, heads, sites, head_dim)`` that ``nlp_create_qkv_heads`` emits rather than the 6-D
    volume -- both shapes are accepted, because refusing the flat one refuses the whole fast path.
    The op is SITE-major, so at more than one head per chip the flat form is transposed here
    (see ``as_volume``); at one head the two layouts are the same bytes.

    ``already_bricked``: sites are already in bricked order (a caller-side hoist). Q/K/V are
    site-major buffers labelled ``(batch, heads, bricked_sites, head_dim)`` -- read as
    ``(batch, 1, sites, heads * head_dim)`` -- the W halo is ``neighbor_pad`` on ``W_br``, and
    the return stays bricked. ``brick`` is then required so the caller and the op cannot disagree.

    K and V are halo-exchanged in BRICKED order on both paths: brick the owned columns first, then
    pad ``W_br`` by whole bricks. That is the same tensor as bricking the widened natural volume
    (the planner requires ``brick_w | width_local``, so no brick straddles a shard seam), and the
    stick is ``32 * channels`` wide, clear of the 128 B width that hangs ``neighbor_pad``. The
    natural-order exchange with its W-fold that this replaced is what kept odd brick widths out.

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
        assert brick is not None, "already_bricked needs the brick the stage converted with"
    elif len(query.shape) == 4:
        # Flat HEAD-major (batch, heads, sites, head_dim); sites run (t, h, w_local). Turned into
        # the site-major volume by ``as_volume`` below, not by a reshape: that was a view only at
        # one head per chip and interleaved heads with sites at any other count.
        batch, head_count, _, head_dim = tuple(query.shape)
    else:
        batch, time_extent, height_extent, width_local, head_count, head_dim = tuple(query.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"

    volume = dims
    assert (
        volume[2] == width_local * shard_count
    ), f"W {volume[2]} does not split into {shard_count} shards of {width_local}"

    context_window = tuple(min(window, extent) for window, extent in zip(kernel_size, volume))
    if stride is None:
        stride = (1, 1, 1)
    if brick is None:
        brick = brick_override(volume) or _choose_sharded_brick(
            volume, context_window, stride, width_local, shard_count
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
    plan = cached_bricked_plan(
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

    t_br, h_br, w_br = brick_grid(owned_volume, brick)

    def as_volume(tensor: ttnn.Tensor, lane: str) -> ttnn.Tensor:
        """``(b, heads, sites, hd)`` TILE or ``(b, t, h, w, heads, hd)`` -> ``(b, t, h, w_local, C)`` ROW_MAJOR.

        The flat form is HEAD-major (what ``nlp_create_qkv_heads`` emits); the op is SITE-major, a
        site's heads being its channels. With more than one head per chip the two differ by a real
        transpose, paid here once per lane. With one head they are the same bytes and it is a view.
        Until 2026-09-11 the flat form was reshaped straight into the volume, which is right only
        at one head per chip (stage 5 under TP4) and silently interleaves heads and sites otherwise.
        """
        with _deep_prof(device, f"{lane}: untilize", category=decode_tree.RESHAPE):
            rows = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        if len(rows.shape) == 4 and head_count > 1:
            with _deep_prof(device, f"{lane}: heads-to-sites", category=decode_tree.RESHAPE):
                moved = ttnn.permute(rows, (0, 2, 1, 3))
            if rows is not tensor:
                ttnn.deallocate(rows)
            rows = moved
        return ttnn.reshape(rows, (batch, time_extent, height_extent, width_local, channels))

    def exchange(grid5: ttnn.Tensor, lane: str) -> ttnn.Tensor:
        """Halo-exchange a ``(b, T_br, H_br, W_br, 32*C)`` ROW_MAJOR brick grid on ``W_br`` -> op layout.

        One brick of sites is folded into the stick, so the stick is ``32 * channels`` wide -- never
        near the 128 B width that hangs neighbor_pad in natural order -- and the halo is whole
        bricks, the unit the planner addresses. The stick has an UPPER bound too, and it is low:
        above 4 KB it is cut into sub-columns of the same memory (``_halo_split``) and the pad
        scales with the cut.
        """
        halo_br = halo // brick[2]
        parts = _halo_split(SITES_PER_BRICK * channels, grid5.element_size())
        _tp_trace(
            device,
            f"{lane}: about to neighbor_pad halo_br={halo_br} links={num_links} parts={parts} "
            f"stick={SITES_PER_BRICK * channels // parts * grid5.element_size()}B",
        )
        with _deep_prof(device, f"{lane}: halo-exchange", category=decode_tree.ALLGATHER):
            split = ttnn.reshape(grid5, (batch, t_br, h_br, w_br * parts, SITES_PER_BRICK * channels // parts))
            exchanged = _halo_exchange(
                ccl_manager,
                split,
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

    def widened_bricked(tensor: ttnn.Tensor, lane: str = "?") -> ttnn.Tensor:
        """K/V halo for the hoisted path: the sites are already bricked, so this is a reshape into
        the brick grid and the exchange -- no 7-D permute."""
        _tp_trace(device, f"{lane}: untilize in (already_bricked, channels={channels})")
        with _deep_prof(device, f"{lane}: untilize", category=decode_tree.RESHAPE):
            rows = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        grid5 = ttnn.reshape(rows, (batch, t_br, h_br, w_br, SITES_PER_BRICK * channels))
        return exchange(grid5, lane)

    def widened(tensor: ttnn.Tensor, lane: str = "?") -> ttnn.Tensor:
        """This chip's shard plus a halo of each neighbour's edge, in op layout. K and V only:
        Q is bricked over the owned region alone and never comes through here.

        Brick FIRST, exchange SECOND. Bricking the owned columns and then padding ``W_br`` by
        whole bricks gives the same tensor as bricking the widened natural volume: the planner
        requires ``brick_w | width_local`` so no brick straddles a seam, ``W_br`` is the fastest
        brick axis so the neighbours' slabs concatenate into the resident grid, and the T/H ghost
        padding is the same on every shard. Doing it in this order is what lets the exchange run
        on a ``32 * channels`` stick at ANY brick width. A natural-order exchange would have to fold
        W columns into the stick to clear 128 B, and an odd halo cannot fold -- which would exclude
        odd brick widths, the only legal ones at W_local = 15.

        Spans stay split (untilize / brick-permute / halo-exchange / tilize) so the collective and
        the reorder can be read apart; they have different fixes.
        """
        volume_form = as_volume(tensor, lane)
        with _deep_prof(device, f"{lane}: brick-permute", category=decode_tree.RESHAPE):
            grid5 = to_bricked_grid(volume_form, volume=owned_volume, brick=brick)
        _tp_trace(device, f"{lane}: to_bricked_grid done -> {tuple(grid5.shape)}")
        return exchange(grid5, lane)

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
            volume_form = as_volume(query, "q")
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
            compute_kernel_config=_compute_kernel_config(),
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
            _tp_trace(device, f"entering TP block (heads per chip={head_count})")
            # The buffer is site-major with this chip's heads inside each site: (sites, heads, hd).
            # The gather wants head-major (heads, sites, hd) so that dim=1 concatenates whole heads
            # in device order, [chip0 heads | chip1 heads | ...] = global head order. With one head
            # per chip (stage 5 under TP4) the two layouts are the same bytes and the reshape is a
            # view; with more heads per chip (the deterministic stages: 4/2/2 at TP4) it is a real
            # transpose -- reshaping instead would silently interleave heads and sites, which is what
            # the assert this replaced was guarding against.
            if head_count == 1:
                flat = ttnn.reshape(owned, (batch, head_count, sites_local, head_dim))
            else:
                by_site = ttnn.reshape(owned, (batch, sites_local, head_count, head_dim))  # a view
                flat = ttnn.permute(by_site, (0, 2, 1, 3))  # (b, heads, sites, hd), new buffer
            _tp_trace(device, f"head-major -> {tuple(flat.shape)}")
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
