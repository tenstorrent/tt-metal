# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The three device executors of 3D neighborhood attention in the LTX-2.5 DiffVAE decoder.

Same contract for all three: ``q``/``k``/``v`` are ``(B, T, H, W, num_heads, head_dim)``, already
RMS-normed and RoPE'd with Q pre-scaled, and the return is ``(B, T, H, W, num_heads * head_dim)`` in
ROW_MAJOR. Their plans (everything that depends on the geometry and nothing on the weights) live in
``neighborhood_attention_plan``.

* ``neighborhood_attention_3d_linear_order``: tokens in natural row-major order. Gathers each
  query group's key span with ``ttnn.embedding`` and runs dense masked SDPA over it, optionally
  splitting the query work across the mesh with K/V replicated. What stage 1 (index 0) runs, and
  the replicated oracle the sharded tests compare against.

* ``neighborhood_attention_3d_bricked``: the ``neighborhood_scaled_dot_product_attention`` device
  op over the whole volume on every chip. The op consumes tokens BRICKED: 32 consecutive sites are
  a compact 3D box, so one tile row is one brick of video. See ``neighborhood_permute``.

* ``neighborhood_attention_3d_bricked_w_sharded``: the same op over this chip's W-shard. K and V
  are halo-exchanged because a window reaches past the shard seam; Q is not, because the halo's
  queries belong to the neighbour, which computes them itself. The op is told the difference
  through ``query_extent``/``query_origin`` and addresses two brick grids: the resident one for
  K, V and the gather, the owned one for Q and the output.

  With ``already_bricked=True`` (the keep-bricked path) Q/K/V arrive in bricked site order, this
  call only halo-exchanges K/V on the ``W_br`` axis, and the return stays bricked.

Callers select one by name through :class:`NAKernel` (``resolve_na_kernel``) and run it through
:func:`neighborhood_attention_3d`, which routes to the executor the record names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import ttnn

from ..utils import timing_tree
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

# Cap on the elements gathered for one attention call's K and V together, per chip. Bounds peak
# device memory independently of the grid.
DEFAULT_CHUNK_BUDGET = 2**29


@dataclass(frozen=True)
class NAKernel:
    """Which NA3D executor a stage runs, and the layout decisions that follow from it."""

    #: The backend string callers select with (``DiffVAEOptions.stage5_backend`` / ``stages_backend``).
    name: str
    #: Keep this chip's W-shard of the sequence through the whole stage.
    w_sharded: bool = False
    #: Sites in bricked order, one tile row per 3D brick.
    bricked: bool = False
    #: Convert to bricked order once at stage entry and back at exit instead of per block.
    keep_bricked: bool = False


NA_KERNELS: dict[str, NAKernel] = {
    kernel.name: kernel
    for kernel in (
        NAKernel("linear_order"),
        NAKernel("bricked", bricked=True),
        NAKernel("bricked_sp_w_sharded", w_sharded=True, bricked=True, keep_bricked=True),
    )
}


def resolve_na_kernel(backend: str | NAKernel) -> NAKernel:
    """The kernel record for a backend name. Rejects an unknown one at construction."""
    if isinstance(backend, NAKernel):
        return backend
    try:
        return NA_KERNELS[backend]
    except KeyError:
        msg = f"unknown NA3D backend {backend!r}; expected one of {sorted(NA_KERNELS)}"
        raise ValueError(msg) from None


def neighborhood_attention_3d(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel: str | NAKernel,
    kernel_size: tuple[int, int, int],
    scale: float = 1.0,
    dims: tuple[int, int, int] | None = None,
    ccl_manager=None,
    sp_axis: int | None = None,
    tp_axis: int | None = None,
    heads_presharded: bool = False,
    brick: tuple[int, int, int] | None = None,
    stride: tuple[int, int, int] | None = None,
    device_plan: NA3DDevicePlan | None = None,
) -> ttnn.Tensor:
    """Run the executor ``kernel`` names, with the arguments that executor understands.

    Q/K/V arrive already RMS-normed, RoPE'd and (for Q) pre-scaled, so ``scale`` is 1.0. The
    replicated kernels return the full volume on every chip; the W-sharded one returns this chip's
    band and needs ``dims`` (the FULL grid), ``sp_axis`` and ``ccl_manager``. ``brick`` set means
    the sites are already in bricked order (keep-bricked). ``device_plan`` is the linear-order
    executor's precomputed plan; ``stride`` is the GNA query-group stride, physical ``(t, h, w)``.
    """
    kernel = resolve_na_kernel(kernel)
    if kernel.w_sharded:
        assert dims is not None, f"{kernel.name} needs the full dims"
        return neighborhood_attention_3d_bricked_w_sharded(
            q,
            k,
            v,
            dims=dims,
            kernel_size=kernel_size,
            sp_axis=sp_axis,
            ccl_manager=ccl_manager,
            scale=scale,
            tp_axis=tp_axis,
            heads_presharded=heads_presharded,
            already_bricked=brick is not None,
            brick=brick,
            stride=stride,
        )
    if kernel.bricked:
        return neighborhood_attention_3d_bricked(q, k, v, kernel_size=kernel_size, scale=scale, stride=stride)
    return neighborhood_attention_3d_linear_order(
        q,
        k,
        v,
        kernel_size=kernel_size,
        scale=scale,
        device_plan=device_plan,
        ccl_manager=ccl_manager,
        gna_stride=stride,
    )


def _compute_kernel_config() -> ttnn.WormholeComputeKernelConfig:
    """HiFi2 with an exact exp, matching the general SDPA op the replicated reference runs.

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

    One gather per mesh axis, tile axis first, then row axis: the order :func:`_emitted_order` assumes.
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
    """3D neighborhood attention with tokens in natural order: gather, then dense masked SDPA.

    Pass ``scale=1.0`` when the caller has pre-scaled Q. Either input layout is accepted.
    ``chunk_budget`` caps the elements gathered per attention call; a group's tiles are
    independent, so splitting the batch is exact. When ``device_plan`` is sharded each chip
    evaluates a slice of every group and the results are gathered back, so the return is the
    same full volume on every chip either way. ``ccl_manager`` is only consulted when this builds
    its own plan.

    ``gna_stride`` is refused unless trivial: this executor has no stride parameter, and silently
    ignoring one is how a caller ends up measuring standard NA and reporting it as GNA.
    """
    assert gna_stride in (None, (1, 1, 1)), (
        f"the linear-order executor has no stride parameter, so gna_stride={gna_stride} would be ignored; "
        f"use a bricked executor, which takes it directly"
    )

    batch, t, h, w, heads, head_dim = tuple(q.shape)
    assert batch == 1, f"batched NA3D is not implemented; got batch={batch}"
    # ttnn.embedding requires a bfloat16 table. Not cast: a quiet downcast would make an fp32
    # caller think it had fp32 attention.
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
    # ttnn.embedding can gather rows out of; a pure stride change in ROW_MAJOR.
    width = heads * head_dim
    tables = [ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (t * h * w, width)) for x in (q, k, v)]

    outputs = []
    for group in device_plan.groups:
        n_tiles = group.local_tiles
        per_tile = group.n_keys * width
        tiles_per_chunk = max(1, min(n_tiles, chunk_budget // max(1, 2 * per_tile)))

        chunks = []
        for start in range(0, n_tiles, tiles_per_chunk):
            tiles = min(tiles_per_chunk, n_tiles - start)
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
                rows = ttnn.embedding(index, table, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=q.dtype)
                rows = ttnn.reshape(rows, (tiles, count, heads, head_dim))
                rows = ttnn.permute(rows, (0, 2, 1, 3))
                gathered.append(ttnn.to_layout(rows, ttnn.TILE_LAYOUT))
            # Never deallocated: on the single-chunk path these ARE the cached plan's tensors.
            del chunk_indices

            # The mask is [1, 1, Nq, Nk] and broadcasts over the tile batch.
            attended = ttnn.transformer.scaled_dot_product_attention(
                gathered[0], gathered[1], gathered[2], attn_mask=group.mask, is_causal=False, scale=1.0
            )
            for tensor in gathered:
                ttnn.deallocate(tensor)

            attended = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
            attended = ttnn.permute(attended, (0, 2, 1, 3))
            chunks.append(ttnn.reshape(attended, (tiles, group.local_queries, width)))

        # Chunks are joined before the gather: chunking is a local memory decision and must not
        # reach the fabric, or the reassembled order would depend on it.
        local = chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=0)
        for tensor in chunks:
            if tensor is not local:
                ttnn.deallocate(tensor)
        outputs.append(ttnn.reshape(local, (group.local_tiles * group.local_queries, width)))

    # One stack per chip, then one gather per mesh axis for the whole call.
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

    # PHYSICAL (t, h, w), passed to the op unpermuted. (1,1,1) is the shipped architecture.
    if stride is None:
        stride = (1, 1, 1)

    # DIFFVAE_NA_WINDOW overrides the architectural window. It changes what the model sees, so it
    # is an experiment knob, not a default.
    window_env = os.environ.get("DIFFVAE_NA_WINDOW")
    if window_env:
        kernel_size = tuple(int(part) for part in window_env.split(","))

    volume = (time_extent, height_extent, width_extent)
    # An axis shorter than the window is attended to in full, matching the op's own clamping.
    context_window = tuple(min(window, extent) for window, extent in zip(kernel_size, volume))
    brick = brick_override(volume) or tuple(ttnn.transformer.neighborhood_choose_brick(context_window))
    if scale is None:
        scale = head_dim**-0.5

    device = query.device()
    plan = cached_bricked_plan(volume, context_window, stride, brick, device)
    bricked_sites = brick_count(volume, brick) * SITES_PER_BRICK
    channels = head_count * head_dim

    def to_op_layout(tensor: ttnn.Tensor) -> ttnn.Tensor:
        """(B,T,H,W,heads,head_dim) -> (B, 1, bricked_sites, heads*head_dim) in TILE.

        The op reads site-major, so heads never move: the 3D brick reorder is the only data movement.
        """
        rows = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        volume_form = ttnn.reshape(rows, (batch, time_extent, height_extent, width_extent, channels))
        bricked = to_bricked(volume_form, volume=volume, brick=brick)
        site_major = ttnn.reshape(bricked, (batch, 1, bricked_sites, channels))
        return ttnn.to_layout(site_major, ttnn.TILE_LAYOUT)

    with timing_tree.span(device, "brick-permute (q,k,v)", category=timing_tree.RESHAPE, deep=True):
        query_op = to_op_layout(query)
        key_op = to_op_layout(key)
        value_op = to_op_layout(value)

    _tp_trace(
        device,
        f"about to call the op: chunk={plan['query_chunk_bricks']} "
        f"gather={plan['gather_brick_count']} kv_tiles={_tiles_per_kv_chunk(plan['gather_brick_count'])}",
    )
    with timing_tree.span(device, "neighborhood-sdpa", category=timing_tree.SDPA, deep=True):
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

    with timing_tree.span(device, "unbrick-permute", category=timing_tree.RESHAPE, deep=True):
        rows = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
        merged = ttnn.reshape(rows, (batch, bricked_sites, channels))
        natural = to_natural(merged, volume=volume, brick=brick)

    return natural


def _tp_trace(device, message: str) -> None:
    """Under ``DIFFVAE_TP_TRACE=1``, synchronise and log, so a hang names the op it hung on. Each
    call is a full mesh sync."""
    if os.environ.get("DIFFVAE_TP_TRACE") != "1":
        return
    from loguru import logger

    ttnn.synchronize_device(device)
    logger.info(f"[tp-trace] {message}")


# The widest stick ``neighbor_pad_async`` moves intact, in bytes. Measured, not documented by the
# op: above this the tail of every halo column is left unwritten, silently, with the right shape
# (models/tt_dit/tests/unit/test_halo_exchange_geometry.py::test_halo_exchange_moves_whole_sticks).
MAX_HALO_STICK_BYTES = 4096


def _halo_split(stick_elements: int, element_bytes: int) -> int:
    """How many sub-columns to cut one halo stick into so the exchange stays inside the bound.

    A ``[.., w_br, s]`` brick grid is contiguous, so ``[.., w_br * parts, s / parts]`` is the SAME
    memory and padding ``parts``-scaled columns pads the same halo. Returns 1 when the stick fits.
    """
    parts = 1
    while stick_elements // parts * element_bytes > MAX_HALO_STICK_BYTES and (stick_elements // parts) % 2 == 0:
        parts *= 2
    return parts


def _halo_exchange(ccl_manager, tensor, *, dims, pad_left, pad_right, axes, neighbor_sems, num_links):
    """neighbor_pad with the topology pinned to Linear, independent of the manager's setting.

    ``neighbor_pad_async`` deadlocks on ``Topology.Ring``. Every other collective in the decode is
    a ring-safe all_gather, so ring stays on for them and only this call drops to Linear.
    ``DIFFVAE_NA_HALO_TOPOLOGY=ring`` hands it back the manager's topology, to retest.
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

    Each chip widens its K/V shard by a halo on both sides, runs the op with window placement kept
    GLOBAL, and computes only the queries it owns. Clamping therefore happens at the true volume
    boundary rather than at a shard seam. The halo is symmetric because every device runs one
    program and must hold the same resident extent; the device at the low edge then sits at a
    negative origin, which the op's signed shard origin handles.

    ``tp_axis`` adds tensor parallelism over heads on an orthogonal mesh axis: each chip keeps
    ``heads/tp`` and they are all-gathered back after the attention. Q/K/V may arrive as the 6-D
    volume or as the flat HEAD-major ``(batch, heads, sites, head_dim)`` that
    ``nlp_create_qkv_heads`` emits (``heads_presharded``); the op is SITE-major, so the flat form
    is transposed in ``as_volume``.

    ``already_bricked``: sites are already in bricked order (the keep-bricked path). Q/K/V are
    site-major buffers labelled ``(batch, heads, bricked_sites, head_dim)``, the W halo is
    ``neighbor_pad`` on ``W_br``, and the return stays bricked. ``brick`` is then required.

    K and V are halo-exchanged in BRICKED order on both paths: brick the owned columns first, then
    pad ``W_br`` by whole bricks. The planner requires ``brick_w | width_local`` so no brick
    straddles a seam, and the stick is ``32 * channels`` wide at any brick width.

    ``stride`` is the GNA query-group stride in PHYSICAL (t, h, w) sites, defaulting to (1,1,1).
    A caller that derives its own brick must pass the same stride here.
    """
    shard_count = int(list(query.device().shape)[sp_axis])
    time_extent, height_extent = dims[0], dims[1]
    width_local = dims[2] // shard_count
    if already_bricked:
        batch, head_count, _, head_dim = tuple(query.shape)
        assert brick is not None, "already_bricked needs the brick the stage converted with"
    elif len(query.shape) == 4:
        # Flat HEAD-major (batch, heads, sites, head_dim); sites run (t, h, w_local).
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
    # K and V span the resident region (owned + halo); Q and the output span only the owned columns.
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
    # DIFFVAE_NA_HALO_LINKS overrides the link count for this halo exchange only.
    num_links = int(os.environ.get("DIFFVAE_NA_HALO_LINKS", 0)) or max(1, ccl_manager.num_links)
    semaphore = ccl_manager.get_np_ping_pong_semaphore(sp_axis)

    t_br, h_br, w_br = brick_grid(owned_volume, brick)

    def as_volume(tensor: ttnn.Tensor, lane: str) -> ttnn.Tensor:
        """``(b, heads, sites, hd)`` TILE or ``(b, t, h, w, heads, hd)`` -> ``(b, t, h, w_local, C)`` ROW_MAJOR.

        The flat form is HEAD-major and the op is SITE-major, so with more than one head per chip
        the two differ by a real transpose. A reshape is right only at one head per chip and
        silently interleaves heads and sites otherwise.
        """
        with timing_tree.span(device, f"{lane}: untilize", category=timing_tree.RESHAPE, deep=True):
            rows = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        if len(rows.shape) == 4 and head_count > 1:
            with timing_tree.span(device, f"{lane}: heads-to-sites", category=timing_tree.RESHAPE, deep=True):
                moved = ttnn.permute(rows, (0, 2, 1, 3))
            if rows is not tensor:
                ttnn.deallocate(rows)
            rows = moved
        return ttnn.reshape(rows, (batch, time_extent, height_extent, width_local, channels))

    def exchange(grid5: ttnn.Tensor, lane: str) -> ttnn.Tensor:
        """Halo-exchange a ``(b, T_br, H_br, W_br, 32*C)`` ROW_MAJOR brick grid on ``W_br`` -> op layout.

        The stick is one brick of sites (``32 * channels``), and the halo is whole bricks. Above
        the stick bound it is cut into sub-columns of the same memory (``_halo_split``).
        """
        halo_br = halo // brick[2]
        parts = _halo_split(SITES_PER_BRICK * channels, grid5.element_size())
        _tp_trace(
            device,
            f"{lane}: about to neighbor_pad halo_br={halo_br} links={num_links} parts={parts} "
            f"stick={SITES_PER_BRICK * channels // parts * grid5.element_size()}B",
        )
        with timing_tree.span(device, f"{lane}: halo-exchange", category=timing_tree.ALLGATHER, deep=True):
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
        with timing_tree.span(device, f"{lane}: tilize", category=timing_tree.RESHAPE, deep=True):
            site_major = ttnn.reshape(exchanged, (batch, 1, bricked_sites, channels))
            out = ttnn.to_layout(site_major, ttnn.TILE_LAYOUT)
        _tp_trace(device, f"{lane}: tilized -> {tuple(out.shape)}")
        return out

    def widened_bricked(tensor: ttnn.Tensor, lane: str = "?") -> ttnn.Tensor:
        """K/V halo for the keep-bricked path: a reshape into the brick grid and the exchange."""
        _tp_trace(device, f"{lane}: untilize in (already_bricked, channels={channels})")
        with timing_tree.span(device, f"{lane}: untilize", category=timing_tree.RESHAPE, deep=True):
            rows = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        grid5 = ttnn.reshape(rows, (batch, t_br, h_br, w_br, SITES_PER_BRICK * channels))
        return exchange(grid5, lane)

    def widened(tensor: ttnn.Tensor, lane: str = "?") -> ttnn.Tensor:
        """This chip's K or V shard plus a halo of each neighbour's edge, in op layout.

        Brick FIRST, exchange SECOND: bricking the owned columns and then padding ``W_br`` by
        whole bricks gives the same tensor as bricking the widened natural volume, because no
        brick straddles a seam and ``W_br`` is the fastest brick axis.
        """
        volume_form = as_volume(tensor, lane)
        with timing_tree.span(device, f"{lane}: brick-permute", category=timing_tree.RESHAPE, deep=True):
            grid5 = to_bricked_grid(volume_form, volume=owned_volume, brick=brick)
        _tp_trace(device, f"{lane}: to_bricked_grid done -> {tuple(grid5.shape)}")
        return exchange(grid5, lane)

    widen = widened_bricked if already_bricked else widened
    with timing_tree.span(device, "halo+brick-permute (k,v)", category=timing_tree.RESHAPE, deep=True):
        key_op = widen(key, "k")
        value_op = widen(value, "v")

    # Q is NOT widened: the halo's queries belong to the neighbour, and the op is told so via
    # query_extent/query_origin below.
    with timing_tree.span(device, "q-to-seq", category=timing_tree.RESHAPE, deep=True):
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

    with timing_tree.span(device, "neighborhood-sdpa", category=timing_tree.SDPA, deep=True):
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
            # Representative only: each device reads its own origin out of the sharded table.
            shard_origin=(0, 0, -halo),
            # Uniform across the mesh, so compile-time; one program serves every shard.
            query_extent=plan["query_extent"],
            query_origin=plan["query_origin"],
            head_count=head_count,
            scale=scale,
            tiles_per_kv_chunk=_tiles_per_kv_chunk(plan["gather_brick_count"]),
            compute_kernel_config=_compute_kernel_config(),
        )

    _tp_trace(device, f"op returned -> {tuple(attended.shape)}")
    with timing_tree.span(device, "unbrick-permute", category=timing_tree.RESHAPE, deep=True):
        rows = ttnn.to_layout(attended, ttnn.ROW_MAJOR_LAYOUT)
        merged = ttnn.reshape(rows, (batch, query_bricked_sites, channels))
        # The op wrote only the owned queries, so there is no halo to slice off. The keep-bricked
        # path stays bricked; natural order is restored once at stage exit.
        owned = merged if already_bricked else to_natural(merged, volume=owned_volume, brick=brick)

    if tp_axis is not None:
        # Device order along tp_axis IS head order, so gathering head-major on dim=1 concatenates
        # [chip0 heads | chip1 heads | ...] = global head order.
        with timing_tree.span(device, "head-allgather", category=timing_tree.ALLGATHER, deep=True):
            sites_local = query_bricked_sites if already_bricked else time_extent * height_extent * width_local
            _tp_trace(device, f"entering TP block (heads per chip={head_count})")
            # The buffer is site-major (sites, heads, hd). With one head per chip the head-major
            # form is the same bytes; with more it is a real transpose, and a reshape would
            # silently interleave heads and sites.
            if head_count == 1:
                flat = ttnn.reshape(owned, (batch, head_count, sites_local, head_dim))
            else:
                by_site = ttnn.reshape(owned, (batch, sites_local, head_count, head_dim))
                flat = ttnn.permute(by_site, (0, 2, 1, 3))
            _tp_trace(device, f"head-major -> {tuple(flat.shape)}")
            # ttnn.reshape can hand back a VIEW over the same buffer, so ``owned`` is freed only
            # after the gather. The gather runs on TILES.
            tiled = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
            _tp_trace(device, "tilized, about to all_gather dim=1")
            gathered = ccl_manager.all_gather(tiled, dim=1, mesh_axis=tp_axis, use_hyperparams=False)
            _tp_trace(device, f"all_gather done -> {tuple(gathered.shape)}")
            if tiled is not flat:
                ttnn.deallocate(tiled)
            if flat is not owned:
                ttnn.deallocate(flat)
            ttnn.deallocate(owned)

        # (b, heads, sites, head_dim) -> (b, 1, sites, heads * head_dim): the out-proj wants the
        # heads folded into the channels of a site, a transpose of the head and site axes.
        with timing_tree.span(device, "head-unflatten", category=timing_tree.RESHAPE, deep=True):
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
            # Tiled on the way out: this branch returns the exact shape the caller's
            # retile short-circuits on, so it reaches the out-proj matmul as-is.
            owned = ttnn.to_layout(owned, ttnn.TILE_LAYOUT)
            _tp_trace(device, f"retilized -> {tuple(owned.shape)}; TP block done")

    return owned
