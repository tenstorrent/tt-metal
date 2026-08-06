# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Spatial-sharding helpers for the HunyuanVideo 1.5 VAE decoder.

The production decoder is not wired to this contract yet.  These helpers keep
the shape/halo rules independent from the decoder graph so they can be
property-tested before every causal convolution and the global mid-block
attention are converted to the Wan/LTX H/W-fractured contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

import ttnn
from models.tt_dit.utils.tensor import typed_tensor_2dshard

TILE_SIZE = 32


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError(f"expected value >= 0 and divisor > 0, got {value=}, {divisor=}")
    return (value + divisor - 1) // divisor


def tile_padded(value: int, tile_size: int = TILE_SIZE) -> int:
    """Round a logical extent up to the TTNN tile grid that actually allocates."""
    if value < 0 or tile_size <= 0:
        raise ValueError(f"expected value >= 0 and tile_size > 0, got {value=}, {tile_size=}")
    return ceil_div(value, tile_size) * tile_size


@dataclass(frozen=True)
class SpatialShard:
    """One rank's equal-size storage region and its logical (unpadded) extent."""

    rank_h: int
    rank_w: int
    h_start: int
    h_stop: int
    w_start: int
    w_stop: int
    logical_h_stop: int
    logical_w_stop: int

    @property
    def logical_height(self) -> int:
        return max(0, self.logical_h_stop - self.h_start)

    @property
    def logical_width(self) -> int:
        return max(0, self.logical_w_stop - self.w_start)


@dataclass(frozen=True)
class SpatialShardPlan:
    """Equal-storage 2D partition with exact crop metadata for uneven H/W.

    H/W are padded by replicating the last logical row/column.  Replication is
    required for Hunyuan's convolution boundary semantics; zero-padding the
    final rank, as Wan does, would change pixels at the real bottom/right edge.
    """

    logical_height: int
    logical_width: int
    height_factor: int
    width_factor: int
    storage_height_per_rank: int | None = None
    storage_width_per_rank: int | None = None

    def __post_init__(self) -> None:
        if min(self.logical_height, self.logical_width, self.height_factor, self.width_factor) <= 0:
            raise ValueError("logical dimensions and mesh factors must all be positive")
        if self.storage_height_per_rank is not None and (
            self.storage_height_per_rank <= 0 or self.storage_height_per_rank * self.height_factor < self.logical_height
        ):
            raise ValueError("height storage override cannot hold the logical height")
        if self.storage_width_per_rank is not None and (
            self.storage_width_per_rank <= 0 or self.storage_width_per_rank * self.width_factor < self.logical_width
        ):
            raise ValueError("width storage override cannot hold the logical width")

    @property
    def local_height(self) -> int:
        return self.storage_height_per_rank or ceil_div(self.logical_height, self.height_factor)

    @property
    def local_width(self) -> int:
        return self.storage_width_per_rank or ceil_div(self.logical_width, self.width_factor)

    @property
    def padded_height(self) -> int:
        return self.local_height * self.height_factor

    @property
    def padded_width(self) -> int:
        return self.local_width * self.width_factor

    def shard(self, rank_h: int, rank_w: int) -> SpatialShard:
        if not (0 <= rank_h < self.height_factor and 0 <= rank_w < self.width_factor):
            raise IndexError(f"rank {(rank_h, rank_w)} outside {(self.height_factor, self.width_factor)}")
        h_start = rank_h * self.local_height
        w_start = rank_w * self.local_width
        return SpatialShard(
            rank_h=rank_h,
            rank_w=rank_w,
            h_start=h_start,
            h_stop=h_start + self.local_height,
            w_start=w_start,
            w_stop=w_start + self.local_width,
            logical_h_stop=min(self.logical_height, h_start + self.local_height),
            logical_w_stop=min(self.logical_width, w_start + self.local_width),
        )

    def shards(self) -> tuple[SpatialShard, ...]:
        return tuple(
            self.shard(rank_h, rank_w) for rank_h in range(self.height_factor) for rank_w in range(self.width_factor)
        )

    def scaled(self, height_scale: int, width_scale: int | None = None) -> "SpatialShardPlan":
        width_scale = height_scale if width_scale is None else width_scale
        if height_scale <= 0 or width_scale <= 0:
            raise ValueError("spatial scales must be positive")
        return SpatialShardPlan(
            self.logical_height * height_scale,
            self.logical_width * width_scale,
            self.height_factor,
            self.width_factor,
            self.local_height * height_scale,
            self.local_width * width_scale,
        )


def plan_supports_rank_local_edge_fill(plan: SpatialShardPlan) -> bool:
    """Whether every rank keeps at least one logical row and column.

    ``canonicalize_replicated_shard_edges`` rebuilds the replicate tail from a
    row/column the same rank already holds, so a rank whose storage is entirely
    padding has nothing to replicate from.  Both production grids satisfy this
    on 8x4: 30 rows over 8 ranks leaves a 2-row tail inside a 4-row shard, and
    45 rows leaves a 3-row tail inside a 6-row shard.
    """
    return (plan.padded_height - plan.logical_height) < plan.local_height and (
        plan.padded_width - plan.logical_width
    ) < plan.local_width


def replicate_pad_to_plan(x: torch.Tensor, plan: SpatialShardPlan, *, h_dim: int = -2, w_dim: int = -1) -> torch.Tensor:
    """Replicate-pad a host tensor to the equal storage shape required by a mesh."""
    h_dim %= x.ndim
    w_dim %= x.ndim
    if h_dim == w_dim:
        raise ValueError("height and width dimensions must differ")
    if x.shape[h_dim] != plan.logical_height or x.shape[w_dim] != plan.logical_width:
        raise ValueError(
            f"tensor H/W {(x.shape[h_dim], x.shape[w_dim])} do not match plan "
            f"{(plan.logical_height, plan.logical_width)}"
        )

    out = x
    if plan.padded_height > plan.logical_height:
        index = [slice(None)] * out.ndim
        index[h_dim] = slice(plan.logical_height - 1, plan.logical_height)
        out = torch.cat(
            [
                out,
                out[tuple(index)].expand(
                    *[
                        plan.padded_height - plan.logical_height if dim == h_dim else size
                        for dim, size in enumerate(out.shape)
                    ]
                ),
            ],
            dim=h_dim,
        )
    if plan.padded_width > plan.logical_width:
        index = [slice(None)] * out.ndim
        index[w_dim] = slice(plan.logical_width - 1, plan.logical_width)
        out = torch.cat(
            [
                out,
                out[tuple(index)].expand(
                    *[
                        plan.padded_width - plan.logical_width if dim == w_dim else size
                        for dim, size in enumerate(out.shape)
                    ]
                ),
            ],
            dim=w_dim,
        )
    return out


def host_shard_with_halo(
    padded: torch.Tensor,
    plan: SpatialShardPlan,
    rank_h: int,
    rank_w: int,
    *,
    halo_h: int,
    halo_w: int,
) -> torch.Tensor:
    """Host oracle for one neighbor-pad exchange on an NCHW tensor.

    This models TTNN ``neighbor_pad`` with ``padding_mode="replicate"``:
    interior halos come from adjacent ranks and only global edges replicate.
    """
    if padded.ndim != 4:
        raise ValueError(f"expected NCHW rank 4, got shape {tuple(padded.shape)}")
    if padded.shape[-2:] != (plan.padded_height, plan.padded_width):
        raise ValueError("padded tensor does not match plan")
    if halo_h < 0 or halo_w < 0:
        raise ValueError("halo widths must be non-negative")

    shard = plan.shard(rank_h, rank_w)
    globally_padded = torch.nn.functional.pad(padded, (halo_w, halo_w, halo_h, halo_h), mode="replicate")
    return globally_padded[
        :,
        :,
        shard.h_start : shard.h_stop + 2 * halo_h,
        shard.w_start : shard.w_stop + 2 * halo_w,
    ]


def stitch_host_shards(shards: list[torch.Tensor], plan: SpatialShardPlan) -> torch.Tensor:
    """Join rank-major equal-storage NCHW shards and crop padding."""
    if len(shards) != plan.height_factor * plan.width_factor:
        raise ValueError(f"expected {plan.height_factor * plan.width_factor} shards, got {len(shards)}")
    rows = []
    for rank_h in range(plan.height_factor):
        start = rank_h * plan.width_factor
        rows.append(torch.cat(shards[start : start + plan.width_factor], dim=-1))
    return torch.cat(rows, dim=-2)[..., : plan.logical_height, : plan.logical_width]


def canonicalize_host_shard_edges(
    shard: torch.Tensor, plan: SpatialShardPlan, rank_h: int, rank_w: int
) -> torch.Tensor:
    """Host oracle for rank-local replicate-edge canonicalization."""
    if shard.shape[-2:] != (plan.local_height, plan.local_width):
        raise ValueError(f"shard shape {tuple(shard.shape[-2:])} does not match plan")
    out = shard.clone()
    logical = plan.shard(rank_h, rank_w)
    if logical.logical_height < plan.local_height:
        out[..., logical.logical_height :, :] = out[..., logical.logical_height - 1 : logical.logical_height, :]
    if logical.logical_width < plan.local_width:
        out[..., :, logical.logical_width :] = out[..., :, logical.logical_width - 1 : logical.logical_width]
    return out


def causal_upsampled_frames(latent_frames: int, temporal_upsample_stages: int) -> int:
    """Hunyuan DCAE's asymmetric temporal expansion: frame zero is never doubled."""
    if latent_frames <= 0 or temporal_upsample_stages < 0:
        raise ValueError("latent_frames must be positive and stages non-negative")
    return 1 + (latent_frames - 1) * (2**temporal_upsample_stages)


@dataclass(frozen=True)
class AttentionChunk:
    """One query block of the VAE mid-block's block-causal attention.

    Every query row in the block belongs to the single latent frame ``frame``,
    so the block reads the whole causal prefix ``[0, kv_stop)`` and needs no
    additive mask.

    ``q_start``/``q_stop`` index the query sequence the caller actually holds,
    which is the global sequence on a replicated rank and the rank-local
    sequence under H/W fracturing.  ``kv_stop`` always indexes the global key
    sequence, because attention reduces over every spatial token of the causal
    prefix regardless of who owns the query.
    """

    frame: int
    q_start: int
    q_stop: int
    kv_stop: int

    @property
    def q_len(self) -> int:
        return self.q_stop - self.q_start

    def score_elements(self, tile_size: int = TILE_SIZE) -> int:
        """Allocated score-block elements, including TTNN tile padding."""
        return tile_padded(self.q_len, tile_size) * tile_padded(self.kv_stop, tile_size)


def block_causal_chunk_plan(
    n_frame: int,
    n_hw: int,
    q_chunk_tokens: int = 0,
    *,
    kv_hw: int | None = None,
) -> tuple[AttentionChunk, ...]:
    """Split block-causal attention queries into mask-free blocks.

    Hunyuan's VAE mask lets a query in latent frame ``f`` attend to every token
    in frames ``0..f``.  Query rows therefore share a key prefix exactly when
    they share a frame, so a block confined to one frame can slice keys and
    values to ``[0, (f + 1) * n_hw)`` and drop the mask completely.  Blocks
    spanning frames would need an additive mask as large as their own score
    block, which is the allocation this split exists to remove.

    ``n_hw`` counts the query tokens one frame contributes to the sequence the
    caller holds.  ``kv_hw`` counts the key tokens one frame contributes to the
    global sequence and defaults to ``n_hw``.  They differ exactly when the
    queries are H/W-fractured across a mesh: a rank owns ``local_h * local_w``
    query rows per frame but must still reduce over all ``logical_h *
    logical_w`` keys of every causal-prior frame.  The rank's query rows are
    contiguous within a frame in its own storage even though they are strided
    in the global sequence, which is why one ``(q_start, q_stop)`` range per
    frame still describes them.

    ``q_chunk_tokens`` of 0 means one block per frame.
    """
    if n_frame <= 0 or n_hw <= 0:
        raise ValueError(f"expected positive n_frame and n_hw, got {n_frame=}, {n_hw=}")
    if q_chunk_tokens < 0:
        raise ValueError(f"q_chunk_tokens must be non-negative, got {q_chunk_tokens}")
    kv_hw = n_hw if kv_hw is None else kv_hw
    if kv_hw <= 0:
        raise ValueError(f"kv_hw must be positive, got {kv_hw}")

    step = n_hw if q_chunk_tokens == 0 else min(q_chunk_tokens, n_hw)
    chunks = []
    for frame in range(n_frame):
        frame_start = frame * n_hw
        kv_stop = (frame + 1) * kv_hw
        for offset in range(0, n_hw, step):
            chunks.append(
                AttentionChunk(
                    frame=frame,
                    q_start=frame_start + offset,
                    q_stop=frame_start + min(offset + step, n_hw),
                    kv_stop=kv_stop,
                )
            )
    return tuple(chunks)


def chunk_plan_peak_score_elements(plan, tile_size: int = TILE_SIZE) -> int:
    """Largest single score block a plan allocates, which bounds peak memory."""
    return max((chunk.score_elements(tile_size) for chunk in plan), default=0)


def unchunked_score_elements(n_frame: int, n_hw: int, tile_size: int = TILE_SIZE) -> int:
    """Allocated elements of the monolithic ``seq x seq`` score or mask tensor."""
    if n_frame <= 0 or n_hw <= 0:
        raise ValueError(f"expected positive n_frame and n_hw, got {n_frame=}, {n_hw=}")
    padded = tile_padded(n_frame * n_hw, tile_size)
    return padded * padded


def attention_chunk_tokens_from_env(environ=None) -> int:
    """Read ``HY_VAE_ATTN_CHUNK``; 0 keeps the monolithic attention default."""
    raw = (os.environ if environ is None else environ).get("HY_VAE_ATTN_CHUNK", "0")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"HY_VAE_ATTN_CHUNK must be an integer token count, got {raw!r}") from error
    if value < 0:
        raise ValueError(f"HY_VAE_ATTN_CHUNK must be non-negative, got {value}")
    return value


def _bool_from_env(name: str, environ=None) -> bool:
    """Strict 0/1 gate: anything else fails closed rather than silently off."""
    raw = (os.environ if environ is None else environ).get(name, "0")
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be '0' or '1', got {raw!r}")
    return raw == "1"


def attention_distributed_from_env(environ=None) -> bool:
    """Read ``HY_VAE_ATTN_DIST``; 0 keeps replicated mid-block attention."""
    return _bool_from_env("HY_VAE_ATTN_DIST", environ)


def attention_sdpa_from_env(environ=None) -> bool:
    """Read ``HY_VAE_ATTN_SDPA``; 0 keeps the explicit matmul/softmax blocks."""
    return _bool_from_env("HY_VAE_ATTN_SDPA", environ)


# ---------------------------------------------------------------------------
# Flash-SDPA geometry budget
# ---------------------------------------------------------------------------

# Blackhole gives each Tensix core 1.5 MiB of L1, part of which is reserved for
# kernel binaries, semaphores, and runtime args.  Circular buffers must fit in
# what is left, so plan against a deliberately conservative fraction.
BLACKHOLE_L1_BYTES = 1536 * 1024
SDPA_L1_BUDGET_BYTES = 1024 * 1024
BF16_TILE_BYTES = 2 * TILE_SIZE * TILE_SIZE


def sdpa_cb_tiles(q_chunk: int, k_chunk: int, head_dim: int) -> int:
    """Upper bound on the circular-buffer tiles ttnn's flash SDPA allocates.

    Mirrors the CB sizing in
    ``ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp``
    (the ``q_tiles``/``k_tiles``/``v_tiles``/``qk_tiles``/``out_im_tiles``/
    ``out0_t`` block).  Q is assumed double buffered and ``cb_out`` is charged
    at its non-streaming size, so this over-counts rather than under-counts.

    The ``k_tiles``/``v_tiles`` terms are ``Sk_chunk_t * DHt * 2`` each, which
    is why a head dim of 1024 (``DHt = 32``) makes the key chunk size, not the
    sequence length, the binding constraint.
    """
    for name, value in (("q_chunk", q_chunk), ("k_chunk", k_chunk), ("head_dim", head_dim)):
        if value <= 0 or value % TILE_SIZE:
            raise ValueError(f"{name} must be a positive multiple of {TILE_SIZE}, got {value}")
    sq_chunk_t = q_chunk // TILE_SIZE
    sk_chunk_t = k_chunk // TILE_SIZE
    dht = head_dim // TILE_SIZE
    q_tiles = sq_chunk_t * dht * 2
    k_tiles = sk_chunk_t * dht * 2
    v_tiles = sk_chunk_t * dht * 2
    mask_tiles = 2  # lightweight palette: neginf tile + partial-K tile
    qk_tiles = sq_chunk_t * sk_chunk_t
    out_im_tiles = sq_chunk_t * dht
    out_tiles = sq_chunk_t * dht
    statistics_tiles = sq_chunk_t
    scale_tiles = 1
    return (
        q_tiles + k_tiles + v_tiles + mask_tiles + qk_tiles + out_im_tiles + out_tiles + statistics_tiles + scale_tiles
    )


def sdpa_cb_l1_bytes(q_chunk: int, k_chunk: int, head_dim: int) -> int:
    return sdpa_cb_tiles(q_chunk, k_chunk, head_dim) * BF16_TILE_BYTES


def sdpa_chunks_fit_l1(q_chunk: int, k_chunk: int, head_dim: int, budget: int = SDPA_L1_BUDGET_BYTES) -> bool:
    return sdpa_cb_l1_bytes(q_chunk, k_chunk, head_dim) <= budget


def largest_sdpa_k_chunk(head_dim: int, q_chunk: int = TILE_SIZE, budget: int = SDPA_L1_BUDGET_BYTES) -> int:
    """Biggest tile-aligned key chunk whose circular buffers fit the budget.

    Returns 0 when even a single key tile does not fit, which is the signal
    that the geometry cannot use the flash kernel at all.
    """
    best = 0
    k_chunk = TILE_SIZE
    while sdpa_chunks_fit_l1(q_chunk, k_chunk, head_dim, budget):
        best = k_chunk
        k_chunk += TILE_SIZE
    return best


def _get_edge_mask(logical_h, logical_w, padded_h, padded_w, axis, dtype, parallel_config, ccl_manager):
    """Create one cached rank-selective mask fractured like the activation."""
    cache = getattr(ccl_manager, "_hunyuan_vae_edge_mask_cache", None)
    if cache is None:
        cache = {}
        ccl_manager._hunyuan_vae_edge_mask_cache = cache
    key = (logical_h, logical_w, padded_h, padded_w, axis, dtype)
    if key not in cache:
        mask = torch.zeros((1, 1, padded_h, padded_w, 1), dtype=torch.float32)
        if axis == "h":
            mask[:, :, logical_h:, :, :] = 1
        elif axis == "w":
            mask[:, :, :, logical_w:, :] = 1
        else:
            raise ValueError(f"expected axis 'h' or 'w', got {axis!r}")
        cache[key] = typed_tensor_2dshard(
            mask,
            ccl_manager.mesh_device,
            shard_mapping={
                parallel_config.height_parallel.mesh_axis: 2,
                parallel_config.width_parallel.mesh_axis: 3,
            },
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
        )
    return cache[key]


def _select_masked(original, replacement, mask):
    keep = ttnn.add(ttnn.multiply(mask, -1.0), 1.0)
    return ttnn.add(ttnn.multiply(original, keep), ttnn.multiply(replacement, mask))


def canonicalize_replicated_shard_edges(x_bthwc, logical_h, logical_w, parallel_config, ccl_manager):
    """Restore uneven Hunyuan replicate edges without any collective.

    Every rank builds the same local replacement candidate. Cached masks are
    themselves H/W-fractured, so only storage-only cells on the final H/W
    ranks select that candidate. H is repaired before W, which makes the
    bottom-right corner replicate the true logical corner.
    """
    padded_h = x_bthwc.shape[2] * parallel_config.height_parallel.factor
    padded_w = x_bthwc.shape[3] * parallel_config.width_parallel.factor
    if padded_h == logical_h and padded_w == logical_w:
        return x_bthwc

    out = x_bthwc
    B, T, local_h, local_w, C = out.shape
    h_tail = padded_h - logical_h
    if h_tail:
        valid_h = local_h - h_tail
        if valid_h <= 0:
            ranks = parallel_config.height_parallel.factor
            raise ValueError(
                f"rank-local H edge fill requires at least one logical row on the final rank: "
                f"logical H={logical_h} over {ranks} H ranks stores {local_h} row(s) per rank "
                f"({padded_h} padded), so the last rank holds {h_tail} padding row(s) and nothing "
                f"to replicate from. Use H >= {(ranks - 1) * local_h + 1} at this mesh, or fewer "
                f"H ranks."
            )
        prefix = ttnn.slice(out, [0, 0, 0, 0, 0], [B, T, valid_h, local_w, C])
        edge = ttnn.slice(out, [0, 0, valid_h - 1, 0, 0], [B, T, valid_h, local_w, C])
        replacement = ttnn.concat([prefix] + [edge] * h_tail, dim=2)
        mask = _get_edge_mask(
            logical_h,
            logical_w,
            padded_h,
            padded_w,
            "h",
            out.get_dtype(),
            parallel_config,
            ccl_manager,
        )
        out = _select_masked(out, replacement, mask)

    w_tail = padded_w - logical_w
    if w_tail:
        valid_w = local_w - w_tail
        if valid_w <= 0:
            ranks = parallel_config.width_parallel.factor
            raise ValueError(
                f"rank-local W edge fill requires at least one logical column on the final rank: "
                f"logical W={logical_w} over {ranks} W ranks stores {local_w} column(s) per rank "
                f"({padded_w} padded), so the last rank holds {w_tail} padding column(s) and "
                f"nothing to replicate from. Use W >= {(ranks - 1) * local_w + 1} at this mesh, or "
                f"fewer W ranks."
            )
        prefix = ttnn.slice(out, [0, 0, 0, 0, 0], [B, T, local_h, valid_w, C])
        edge = ttnn.slice(out, [0, 0, 0, valid_w - 1, 0], [B, T, local_h, valid_w, C])
        replacement = ttnn.concat([prefix] + [edge] * w_tail, dim=3)
        mask = _get_edge_mask(
            logical_h,
            logical_w,
            padded_h,
            padded_w,
            "w",
            out.get_dtype(),
            parallel_config,
            ccl_manager,
        )
        out = _select_masked(out, replacement, mask)
    return out


def _blend_ttnn(prior, current, extent: int, *, dim: int, device, dtype):
    extent = min(prior.shape[dim], current.shape[dim], extent)
    if extent == 0:
        return current
    shape = [1] * len(current.shape)
    shape[dim] = extent
    current_weight = ttnn.from_torch(
        (torch.arange(extent, dtype=torch.float32) / extent).reshape(shape),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    prior_weight = ttnn.add(ttnn.multiply(current_weight, -1.0), 1.0)
    begin = [0] * len(current.shape)
    end = list(current.shape)
    end[dim] = extent
    current_overlap = ttnn.slice(current, begin, end)
    begin[dim] = prior.shape[dim] - extent
    end = list(prior.shape)
    prior_overlap = ttnn.slice(prior, begin, end)
    blended = ttnn.add(ttnn.multiply(prior_overlap, prior_weight), ttnn.multiply(current_overlap, current_weight))
    if extent == current.shape[dim]:
        return blended
    begin = [0] * len(current.shape)
    begin[dim] = extent
    return ttnn.concat([blended, ttnn.slice(current, begin, list(current.shape))], dim=dim)


def stitch_tiles_ttnn(
    decoded_tiles_bthwc,
    coords: list[tuple[int, int, int, int]],
    ncol: int,
    blend_h: int,
    blend_w: int,
    row_limit_h: int,
    row_limit_w: int,
    *,
    spatial_scale: int,
    device,
    dtype=ttnn.bfloat16,
):
    """Blend, crop, and stitch a device-resident *replicated* tile batch.

    The input batch dimension is tile-major.  A future H/W-fractured decoder
    will not need this compatibility helper; its output is directly composed
    by spatial mesh dimensions.  For now this removes host postprocessing on a
    single device and provides the exact TTNN primitive sequence needed after
    an eventual tile all-gather.
    """
    if len(coords) == 0 or ncol <= 0 or len(coords) % ncol:
        raise ValueError("coords must form a non-empty rectangular tile grid")
    tiles = []
    for tile_index, (_, _, real_h, real_w) in enumerate(coords):
        begin = [tile_index, 0, 0, 0, 0]
        end = [
            tile_index + 1,
            decoded_tiles_bthwc.shape[1],
            real_h * spatial_scale,
            real_w * spatial_scale,
            decoded_tiles_bthwc.shape[4],
        ]
        tiles.append(ttnn.slice(decoded_tiles_bthwc, begin, end))

    rows = [tiles[start : start + ncol] for start in range(0, len(tiles), ncol)]
    result_rows = []
    for row_index, row in enumerate(rows):
        result_row = []
        for col_index, tile in enumerate(row):
            if row_index > 0:
                tile = _blend_ttnn(rows[row_index - 1][col_index], tile, blend_h, dim=2, device=device, dtype=dtype)
                rows[row_index][col_index] = tile
            if col_index > 0:
                tile = _blend_ttnn(row[col_index - 1], tile, blend_w, dim=3, device=device, dtype=dtype)
                rows[row_index][col_index] = tile
            end = list(tile.shape)
            end[2] = min(end[2], row_limit_h)
            end[3] = min(end[3], row_limit_w)
            result_row.append(ttnn.slice(tile, [0, 0, 0, 0, 0], end))
        result_rows.append(ttnn.concat(result_row, dim=3))
    return ttnn.concat(result_rows, dim=2)
