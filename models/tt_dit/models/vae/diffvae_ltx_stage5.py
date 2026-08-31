# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 DiffVAE decoder stage 5.

Stage 5 denoises patchified noised pixels ``x_t`` under a deterministic context volume
produced by stages 1-4. It is a stack of ``DiffusionNABlock``s -- 3D neighborhood
attention plus a SwiGLU MLP -- modulated by a shared AdaLN-Zero projection of the
timestep embedding. With ``model_output_type="x0"`` and one inference step, a single
pass of :meth:`DiffVAEStage5.forward` returns pixels.

Port of ``ltx_core.model.video_vae``: ``DiffusionVideoDecoder.forward_diff_step`` plus the
``transformer/combined/`` pathway. The 3D neighborhood attention itself is not implemented
here -- see :func:`neighborhood_attention_3d`.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
from loguru import logger

import ttnn

from ...utils import decode_tree

#: Per-stage decode timing. Off unless DIFFVAE_STAGE_TIMING is set (a truthy value), since each
#: probe forces a device sync that would otherwise serialize the async pipeline. The gate lives in
#: decode_tree so this module, na3d and the tree cannot disagree about whether timing is on -- if
#: they did, the tree would hold partial data and every "other" remainder would silently lie.
_STAGE_TIMING = decode_tree.ENABLED


@contextlib.contextmanager
def stage_timer(mesh_device, label: str, *, category: str | None = None, root: bool = False):
    """Sync the mesh and time a decode stage (or its host-side tail). Inert unless DIFFVAE_STAGE_TIMING.

    The label is taken at open rather than on the way out because a tree node names itself from
    birth: a span that never closes is then still identifiable, which is when the name matters most.
    """
    if not _STAGE_TIMING:
        yield
        return
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    span = decode_tree.open_span(label.strip(), category=category, root=root)
    try:
        yield
    except BaseException:
        # No finally: on the raising path this deliberately does not log, exactly as before. The
        # span still has to leave the stack, or every later span nests under a dead parent.
        decode_tree.abort_span(span)
        raise
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - t0) * 1000
    decode_tree.close_span(span, ms)


def deep_prof(mesh_device, key: str, *, category: str | None = None):
    """block_prof, but only under DIFFVAE_BLOCK_PROF.

    These split regions inside one attention call, so they are numerous and individually small
    against the two syncs each one costs. Off unless somebody is chasing exactly this.
    """
    if not decode_tree.DEEP:
        return contextlib.nullcontext()
    return block_prof(mesh_device, key, category=category)


#: Accumulates within-block time by region (attn / mlp / ...) across every block+band, so a single
#: diff-step run reports where the diff-block stack actually goes. Reset per step, reported at its end.
_BLOCK_PROF: dict[str, float] = {}


@contextlib.contextmanager
def block_prof(mesh_device, key: str, *, category: str | None = None):
    """Sync + accumulate a diff-block region's time into _BLOCK_PROF[key]. Inert unless timing is on.
    The syncs serialize the region (so absolute totals inflate a little), but the split is what matters.

    Also records the span in the decode tree. _BLOCK_PROF is still written exactly as before --
    time_diff_block.py and ab_gna_stage5.py import that global directly.
    """
    if not _STAGE_TIMING:
        yield
        return
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    span = decode_tree.open_span(key, category=category)
    try:
        yield
    except BaseException:
        decode_tree.abort_span(span)
        raise
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - t0) * 1000
    decode_tree.close_span(span, ms)
    _BLOCK_PROF[key] = _BLOCK_PROF.get(key, 0.0) + ms


from ...layers.embeddings import TimestepEmbedding, Timesteps
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.na3d import neighborhood_attention_3d as na3d_on_device
from ...layers.na3d import neighborhood_attention_3d_op_sp_w_sharded, window_bounds
from ...layers.neighborhood_permute import (
    SITES_PER_BRICK,
    brick_count,
    brick_grid,
    from_bricked_grid,
    sites_per_t_brick,
    to_bricked_grid,
)
from ...layers.normalization import RMSNorm
from ...utils.tensor import fast_device_to_host
from ...utils.tensor import from_torch as sharded_from_torch
from ...utils.tensor import local_device_to_torch
from ...utils.tensor import to_torch as gathered_to_torch
from ...utils.yuv_d2h import fast_device_to_host_yuv

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DiffVAEStage5",
    "DiffVAEStage5Config",
    "Grid",
    "neighborhood_attention_3d",
    "patchify",
    "unpatchify",
]

# Number of AdaLN-Zero chunks in the shared projection. Only 4 are consumed
# (scale/shift for MSA and MLP); the 3 gate chunks exist for checkpoint shape
# compatibility because static gates were folded into Linear weights at export.
NUM_ADALN_CHUNKS = 7

TILE = 32


class Grid(NamedTuple):
    """Stage-5 volume extent. ``T``/``H``/``W`` are in patch units, not pixels."""

    batch: int
    t: int
    h: int
    w: int

    @property
    def sites(self) -> int:
        return self.t * self.h * self.w


@dataclass(frozen=True)
class DiffVAEStage5Config:
    """Shipped LTX-2.5 DiffVAE stage-5 geometry."""

    dim: int = 256
    head_dim: int = 64
    kernel_size: tuple[int, int, int] = (11, 11, 11)
    # Generalized Neighborhood Attention query-group stride, physical (t,h,w). (1,1,1) is the shipped
    # architecture: standard neighborhood attention, every query centered on its own window. Anything
    # larger shares one window across each group of queries, which the network was not trained for.
    gna_stride: tuple[int, int, int] = (1, 1, 1)
    context_channels: int = 256
    mlp_hidden: int = 1024
    num_blocks: int = 8
    t_emb_dim: int = 384
    patch_size: int = 4
    out_channels: int = 3
    norm_eps: float = 1e-6
    timestep_scale_multiplier: float = 1000.0
    rope_base: float = 10000.0
    rope_dim_split: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.dim % self.head_dim != 0:
            msg = f"dim={self.dim} is not divisible by head_dim={self.head_dim}"
            raise ValueError(msg)

    @property
    def num_heads(self) -> int:
        return self.dim // self.head_dim

    @property
    def patch_channels(self) -> int:
        return self.out_channels * self.patch_size**2

    @property
    def resolved_rope_dim_split(self) -> tuple[int, int, int]:
        if self.rope_dim_split is not None:
            split = self.rope_dim_split
        else:
            split = default_rope_dim_split(self.head_dim)
        if sum(split) != self.head_dim:
            msg = f"rope_dim_split={split} must sum to head_dim={self.head_dim}"
            raise ValueError(msg)
        return split


# ---------------------------------------------------------------------------
# 3D neighborhood attention boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NAKernel:
    """Which NA3D executor stage 5 runs, and the layout decisions that follow from it.

    Everything the W-SP path sets up -- the resharded context, the W-sharded RoPE tables, the local
    W extent, the per-chip row count -- is identical whichever W-sharded backend runs; only the
    attention call differs. Carrying that as a field rather than re-deriving it from the name means
    the dispatch and the shapes cannot disagree, and putting a backend on the sharded path is a
    one-line change in the table below instead of four scattered ones.

    The other three fields used to be independent environment variables, which let combinations
    exist that no executor honours: DIFFVAE_BRICKED_FLAT only ever countermanded
    DIFFVAE_S5_FLAT_SEQ for one backend, and DIFFVAE_S5_KEEP_BRICKED did nothing off the bricked
    path. As fields they are stated once, per kernel, where the reason for each is visible.
    """

    #: The backend string callers select with (DIFFVAE_STAGE5_BACKEND, or the ctor kwarg).
    name: str
    #: Keep this chip's W-shard of the sequence through the whole stage.
    w_sharded: bool = False
    #: Our op: sites in bricked order, one tile row per 3D brick.
    bricked: bool = False
    #: Hoist the brick conversion to stage entry/exit instead of paying it per block.
    keep_bricked: bool = False
    #: Wants the flat (B, NH, S, HD) sequence the projections already produce, rather than the 6-D
    #: volume its to_seq would tear straight back down. Only honoured where the preconditions hold
    #: (see ``_NeighborhoodAttention3D.forward``); the executor asserts them too.
    prefers_flat_seq: bool = False


_NA_KERNELS: dict[str, NAKernel] = {
    kernel.name: kernel
    for kernel in (
        NAKernel("gather"),
        NAKernel("op"),
        NAKernel("fused"),
        NAKernel("op_sp"),
        NAKernel("bricked", bricked=True),
        NAKernel("op_sp_w_sharded", w_sharded=True, prefers_flat_seq=True),
        # No flat sequence: the bricked op re-bricks the sites itself, so the flat form buys it
        # nothing and its 4-D branch has to reconstruct the volume shape anyway.
        NAKernel("bricked_sp_w_sharded", w_sharded=True, bricked=True, keep_bricked=True),
    )
}


def resolve_na_kernel(backend: str | NAKernel) -> NAKernel:
    """The kernel record for a backend name. Rejects an unknown one HERE, at construction.

    Left to itself an unknown name survives the whole build -- weights included -- and surfaces as
    a ValueError from inside na3d's own dispatcher on the first forward.
    """
    if isinstance(backend, NAKernel):
        return backend
    try:
        return _NA_KERNELS[backend]
    except KeyError:
        msg = f"unknown NA3D backend {backend!r}; expected one of {sorted(_NA_KERNELS)}"
        raise ValueError(msg) from None


def neighborhood_attention_3d(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    scale: float = 1.0,
    ccl_manager=None,
    backend: str = "gather",
) -> ttnn.Tensor:
    """3D neighborhood attention over ``(B, T, H, W, num_heads, head_dim)`` tensors.

    Q/K/V arrive already RMS-normed, RoPE'd and (for Q) pre-scaled, so ``scale`` is 1.0.
    Returns ``(B, T, H, W, num_heads * head_dim)``. Mirrors upstream's
    ``NAAttentionCallable`` contract.

    With a ``ccl_manager`` the attention is split across the mesh instead of every chip
    evaluating the whole volume; without one it runs replicated. Either way the result is the
    full volume on every chip, so nothing downstream changes.

    ``backend`` picks the executor: ``"gather"`` (default) is the grouped gather path that the
    ``ccl_manager`` split rides on; ``"op"`` uses the SDPA op's on-device neighborhood mask and
    runs replicated (it ignores ``ccl_manager``), so it is the single-chip / pre-sharding path.

    This was a swap point for a host fallback while the device primitive was being written.
    The dispatch is now direct and unconditional on purpose: a fallback selected by
    ``except ImportError`` would move attention to the host silently, and every parity test
    here would still pass — slower, and no longer measuring the device.
    """
    return na3d_on_device(q, k, v, kernel_size=kernel_size, scale=scale, ccl_manager=ccl_manager, backend=backend)


# ---------------------------------------------------------------------------
# Host-side patch packing
# ---------------------------------------------------------------------------


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Space-to-depth on ``(B, C, T, H, W)``; inverse of :func:`unpatchify`.

    Channel order is ``(c, w_sub, h_sub)`` -- the W sub-index is the *outer* of the two
    spatial ones, matching upstream's ``b c (f p) (h q) (w r) -> b (c p r q) f h w``
    where ``q`` indexes H and ``r`` indexes W.
    """
    batch, channels, t, h, w = x.shape
    p = patch_size
    x = x.reshape(batch, channels, t, h // p, p, w // p, p)
    x = x.permute(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(batch, channels * p * p, t, h // p, w // p)


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Depth-to-space on ``(B, C * patch_size**2, T, H, W)``; inverse of :func:`patchify`."""
    batch, packed, t, h, w = x.shape
    p = patch_size
    channels = packed // (p * p)
    x = x.reshape(batch, channels, p, p, t, h, w)
    x = x.permute(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(batch, channels, t, h * p, w * p)


# ---------------------------------------------------------------------------
# Absolute RoPE
# ---------------------------------------------------------------------------


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    """Split ``head_dim`` across the (T, H, W) RoPE chunks (upstream's default)."""
    if head_dim % 8 != 0:
        msg = f"head_dim={head_dim} must be a multiple of 8 for the default split"
        raise ValueError(msg)
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return (d_t, d_hw, d_hw)


def _rope_inv_freqs(dim: int, base: float) -> torch.Tensor:
    # float64 exponentiation before the float32 cast: upstream computes these in numpy
    # float64, and a float32 pow crosses bf16 rounding boundaries on the high frequencies.
    exponents = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    return (1.0 / base**exponents).to(torch.float32)


class _RopeParts(NamedTuple):
    cos: ttnn.Tensor
    sin: ttnn.Tensor


@dataclass(frozen=True)
class _RopeTables:
    """The fused table as two broadcastable pieces instead of one volume-sized tensor.

    ``frame`` is ``(1, 1, rows_per_frame, head_dim)`` and carries the H and W lanes, which are the
    same in every frame; ``time`` is ``(1, t, 1, head_dim)`` and carries the T lanes, which are the
    same at every site within a frame. See :func:`_build_rope_tables`.
    """

    frame: _RopeParts
    time: _RopeParts
    rows_per_frame: int
    fused: _RopeParts | None = None
    brick: tuple[int, int, int] | None = None
    sites_per_t_br: int = 0

    def frames(self, lo: int, hi: int) -> _RopeTables:
        """The same tables restricted to frames ``[lo, hi)``, for a slab of the volume."""
        if self.fused is not None:
            assert self.brick is not None
            t_br_lo = lo // self.brick[0]
            t_br_hi = _align_up(hi, self.brick[0]) // self.brick[0]
            row_lo = t_br_lo * self.sites_per_t_br
            row_hi = t_br_hi * self.sites_per_t_br
            if (row_lo, row_hi) == (0, self.fused.cos.shape[-2]):
                return self
            return _RopeTables(
                frame=self.frame,
                time=self.time,
                rows_per_frame=self.rows_per_frame,
                fused=_RopeParts(
                    *(ttnn.slice(part, [0, 0, row_lo, 0], [1, 1, row_hi, part.shape[-1]]) for part in self.fused)
                ),
                brick=self.brick,
                sites_per_t_br=self.sites_per_t_br,
            )
        if (lo, hi) == (0, self.time.cos.shape[1]):
            return self
        return _RopeTables(
            frame=self.frame,
            time=_RopeParts(*(ttnn.slice(part, [0, lo, 0, 0], [1, hi, 1, part.shape[-1]]) for part in self.time)),
            rows_per_frame=self.rows_per_frame,
        )


def _rope_pair_swap_matrix(head_dim: int) -> torch.Tensor:
    """``x @ P`` maps adjacent pairs ``(x0, x1)`` to ``(-x1, x0)``.

    A matmul rather than slice-and-concat because the per-axis RoPE chunks start at
    lanes 0/16/40 for head_dim 64 -- none of the odd/even sub-slices land on a tile
    boundary, so every alternative needs a row-major detour per chunk.
    """
    p = torch.zeros(head_dim, head_dim, dtype=torch.float32)
    for j in range(head_dim // 2):
        p[2 * j + 1, 2 * j] = -1.0
        p[2 * j, 2 * j + 1] = 1.0
    return p


def _build_rope_tables(
    grid: Grid,
    *,
    dim_split: tuple[int, int, int],
    base: float,
    num_heads: int,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    w_shard: tuple[int, int] | None = None,
) -> _RopeTables:
    """Fused (T, H, W) absolute RoPE, factored into one frame and one row per frame.

    Upstream rotates in W-slabs of ``rope_num_tiles`` with running absolute offsets, which
    is arithmetically one full-volume rotation over positions ``0..W-1``; the slabbing is
    a Dynamo-shape concern only.

    A row of the fused table is ``[T-lanes(t) | H-lanes(h) | W-lanes(w)]``, so the H and W lanes
    repeat in every frame and the T lanes repeat at every site within a frame. Storing those two
    pieces rather than their combination keeps the table off the critical memory path: the volume
    form is a full activation per table, 9.7 GB each at 6s 1920x1088, and cos plus sin were more
    than half of everything stage 5 held. Each piece is zero outside its own lanes, so
    ``cos == frame.cos + time.cos`` exactly, and :func:`_apply_rope` distributes the multiply over
    the two instead of reassembling the row.
    """
    d_t, d_h, d_w = dim_split
    head_dim = d_t + d_h + d_w
    offsets = (0, d_t, d_t + d_h)

    def lanes(fn, axis: int, positions: torch.Tensor) -> torch.Tensor:
        """``fn`` of this axis's angles, written into this axis's lanes and zero elsewhere.

        Pairs are ``repeat_interleave``d, so pair ``j`` lands on lanes ``(2j, 2j+1)`` of the axis
        chunk with no reordering.
        """
        width = dim_split[axis]
        angles = positions.reshape(-1, 1) * _rope_inv_freqs(width, base).reshape(1, -1)
        rows = torch.zeros(positions.numel(), head_dim, dtype=torch.float32)
        rows[:, offsets[axis] : offsets[axis] + width] = fn(angles).repeat_interleave(2, dim=-1)
        return rows

    within = torch.arange(grid.h * grid.w)
    rows_h = torch.div(within, grid.w, rounding_mode="floor").to(torch.float32)
    rows_w = (within % grid.w).to(torch.float32)
    # Under spatial-W SP the frame piece is split over W (the H/W lanes it carries), so each chip's
    # rows-per-frame is only H*(W/sp)*num_heads; the T-lane ``time`` piece is unaffected by a W-shard.
    if w_shard is not None:
        sp, _ = w_shard
        assert grid.w % sp == 0, f"W={grid.w} must split evenly over sp={sp}"
        rows_per_frame = grid.h * (grid.w // sp) * num_heads
    else:
        rows_per_frame = grid.h * grid.w * num_heads

    def upload(rows: torch.Tensor, shape: tuple[int, ...]) -> ttnn.Tensor:
        return ttnn.from_torch(
            rows.reshape(shape).contiguous(), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype
        )

    def frame_piece(fn) -> ttnn.Tensor:
        # q and k carry heads inside the row axis and the rotation is per-site, so each site's row
        # repeats once per head.
        rows = (lanes(fn, 1, rows_h) + lanes(fn, 2, rows_w)).reshape(grid.h * grid.w, 1, head_dim)
        full = rows.repeat(1, num_heads, 1)  # (H*W, num_heads, head_dim), rows ordered (h, w)
        if w_shard is None:
            return upload(full, (1, 1, grid.h * grid.w * num_heads, head_dim))
        # W-shard: reorder rows to (device, h, w_local, head) so device p gets its W-band, matching the
        # activation's own W-shard (from_torch shards the site dim across sp_axis in device order).
        sp, sp_axis = w_shard
        w_local = grid.w // sp
        reordered = (
            full.reshape(grid.h, sp, w_local, num_heads, head_dim)
            .permute(1, 0, 2, 3, 4)
            .reshape(1, 1, sp * grid.h * w_local * num_heads, head_dim)
        )
        return sharded_from_torch(
            reordered.contiguous(),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_axes=[None, None, sp_axis, None],
        )

    def time_piece(fn) -> ttnn.Tensor:
        return upload(lanes(fn, 0, torch.arange(grid.t, dtype=torch.float32)), (1, grid.t, 1, head_dim))

    return _RopeTables(
        frame=_RopeParts(cos=frame_piece(torch.cos), sin=frame_piece(torch.sin)),
        time=_RopeParts(cos=time_piece(torch.cos), sin=time_piece(torch.sin)),
        rows_per_frame=rows_per_frame,
    )


def _build_bricked_rope_tables(
    grid: Grid,
    brick: tuple[int, int, int],
    *,
    dim_split: tuple[int, int, int],
    base: float,
    num_heads: int,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    w_shard: tuple[int, int] | None = None,
) -> _RopeTables:
    """Fused RoPE in bricked site order, one row per (site, head).

    The factored frame/time form does not survive bricking: T, H and W are interleaved inside
    each 32-site brick. Built once per stage, sliced per band on ``T_br``.
    """
    d_t, d_h, d_w = dim_split
    head_dim = d_t + d_h + d_w
    offsets = (0, d_t, d_t + d_h)
    brick_time, brick_height, brick_width = brick

    def table_for_shard(volume: tuple[int, int, int], w_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(sites * heads, head_dim)`` cos and sin for one W-shard, ghosts zero."""
        t_br, h_br, w_br = brick_grid(volume, brick)
        site = torch.arange(SITES_PER_BRICK)
        dt = torch.div(site, brick_height * brick_width, rounding_mode="floor")
        dh = torch.div(site % (brick_height * brick_width), brick_width, rounding_mode="floor")
        dw = site % brick_width
        t = (torch.arange(t_br).view(t_br, 1, 1, 1) * brick_time + dt).expand(t_br, h_br, w_br, SITES_PER_BRICK)
        h = (torch.arange(h_br).view(1, h_br, 1, 1) * brick_height + dh).expand(t_br, h_br, w_br, SITES_PER_BRICK)
        w = (torch.arange(w_br).view(1, 1, w_br, 1) * brick_width + dw).expand(t_br, h_br, w_br, SITES_PER_BRICK)
        ghost = (t >= volume[0]) | (h >= volume[1]) | (w >= volume[2])
        w = w + w_offset
        t, h, w, ghost = t.reshape(-1), h.reshape(-1), w.reshape(-1), ghost.reshape(-1)

        def lanes(fn, axis: int, positions: torch.Tensor) -> torch.Tensor:
            width = dim_split[axis]
            angles = positions.reshape(-1, 1).to(torch.float32) * _rope_inv_freqs(width, base).reshape(1, -1)
            out = torch.zeros(positions.numel(), head_dim, dtype=torch.float32)
            out[:, offsets[axis] : offsets[axis] + width] = fn(angles).repeat_interleave(2, dim=-1)
            out[ghost] = 0
            return out

        cos = lanes(torch.cos, 0, t) + lanes(torch.cos, 1, h) + lanes(torch.cos, 2, w)
        sin = lanes(torch.sin, 0, t) + lanes(torch.sin, 1, h) + lanes(torch.sin, 2, w)
        return (
            cos.unsqueeze(1).expand(-1, num_heads, -1).reshape(-1, head_dim),
            sin.unsqueeze(1).expand(-1, num_heads, -1).reshape(-1, head_dim),
        )

    if w_shard is not None:
        sp, sp_axis = w_shard
        assert grid.w % sp == 0, f"W={grid.w} must split evenly over sp={sp}"
        w_local = grid.w // sp
        local_volume = (grid.t, grid.h, w_local)
        cos_parts, sin_parts = zip(*(table_for_shard(local_volume, p * w_local) for p in range(sp)))
        fused = _RopeParts(
            cos=sharded_from_torch(
                torch.stack(cos_parts).reshape(1, 1, -1, head_dim).contiguous(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                mesh_axes=[None, None, sp_axis, None],
            ),
            sin=sharded_from_torch(
                torch.stack(sin_parts).reshape(1, 1, -1, head_dim).contiguous(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                mesh_axes=[None, None, sp_axis, None],
            ),
        )
        volume = local_volume
    else:
        cos, sin = table_for_shard((grid.t, grid.h, grid.w), 0)
        fused = _RopeParts(
            cos=ttnn.from_torch(
                cos.reshape(1, 1, -1, head_dim).contiguous(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
            ),
            sin=ttnn.from_torch(
                sin.reshape(1, 1, -1, head_dim).contiguous(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
            ),
        )
        volume = (grid.t, grid.h, grid.w)

    dummy = ttnn.from_torch(torch.zeros(1, 1, 1, head_dim), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    sites_per_t_br = sites_per_t_brick(volume, brick) * num_heads
    return _RopeTables(
        frame=fused,
        time=_RopeParts(cos=dummy, sin=dummy),
        rows_per_frame=sites_per_t_br,
        fused=fused,
        brick=brick,
        sites_per_t_br=sites_per_t_br,
    )


def _apply_rope(
    x: ttnn.Tensor,
    tables: _RopeTables,
    *,
    pair_swap: ttnn.Tensor,
    compute_kernel_config,
) -> ttnn.Tensor:
    """Rotate ``x`` at ``(1, frames, rows_per_frame, head_dim)``. **Consumes** ``x``.

    ``x * cos`` is evaluated as ``x * frame.cos + x * time.cos``, the two pieces broadcasting over
    the frame axis and the row axis respectively. That is two extra multiplies against not
    materialising a table the size of the activation, and it holds no more live tensors than the
    fused form did.

    Bricked-order tables are fused (site order permutes T/H/W together, so the factoring does
    not survive). ``tables.fused`` is then the whole rotation, same shape as ``x``.
    """
    swapped = ttnn.matmul(x, pair_swap, compute_kernel_config=compute_kernel_config)
    if tables.fused is not None:
        aligned = ttnn.multiply(x, tables.fused.cos)
        ttnn.deallocate(x)
        rotated = ttnn.multiply(swapped, tables.fused.sin)
        ttnn.deallocate(swapped)
        return _add_consuming(aligned, rotated)
    aligned = _add_consuming(ttnn.multiply(x, tables.frame.cos), ttnn.multiply(x, tables.time.cos))
    ttnn.deallocate(x)
    rotated = _add_consuming(ttnn.multiply(swapped, tables.frame.sin), ttnn.multiply(swapped, tables.time.sin))
    ttnn.deallocate(swapped)
    return _add_consuming(aligned, rotated)


# ---------------------------------------------------------------------------
# ttnn shape helpers
# ---------------------------------------------------------------------------


def _reshape_retiled(x: ttnn.Tensor, shape: Sequence[int]) -> ttnn.Tensor:
    """Reshape across the tile grid via ROW_MAJOR, where layout is a pure stride change.

    A TILE tensor cannot be reshaped in either of the last two dims without re-tiling,
    which ``ttnn.reshape`` will not do.
    """
    shape = tuple(shape)
    if tuple(x.shape) == shape:
        return x
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.to_layout(ttnn.reshape(rm, shape), ttnn.TILE_LAYOUT)
    # The untilized copy is dead once `out` exists, and at 1920x1088 it is 1.7 GB. Left to
    # refcounting it survives long enough to matter: stage 5 calls this six times per block.
    # Guarded because a ROW_MAJOR input is returned as-is, and that one belongs to the caller.
    if rm is not x:
        ttnn.deallocate(rm)
    return out


def log_dram(mesh_device, label: str) -> None:
    """Log allocated DRAM when ``DIFFVAE_MEM_LOG`` is set.

    Peak memory is what stops this decoder at full resolution, and it is not obvious from the
    source which tensors are still resident: an allocation failure names the op that asked for
    memory, never the ones holding it. This makes the residency curve visible per block.
    """
    if not os.environ.get("DIFFVAE_MEM_LOG"):
        return
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    banks = view.num_banks
    logger.info(
        f"[dram] {label}: allocated {view.total_bytes_allocated_per_bank * banks / 2**30:6.2f} GiB"
        f" of {view.total_bytes_per_bank * banks / 2**30:.2f} GiB,"
        f" largest contiguous free {view.largest_contiguous_bytes_free_per_bank * banks / 2**30:5.2f} GiB"
    )


def _release_intermediates(tensors: Sequence[ttnn.Tensor], *, keep: ttnn.Tensor) -> None:
    """Free every distinct buffer among ``tensors`` except the one ``keep`` is using.

    ``ttnn.reshape`` hands back a NEW Python object that may be a VIEW over its input's buffer,
    and ``ttnn.deallocate`` defaults to ``force=True``. So the ``a is not b`` guard used elsewhere
    in this file is not enough: two distinct objects can name one buffer, and the guard then frees
    memory a live tensor is still reading, or frees the same buffer twice. In _brick_activation
    that returned a stage-5 activation whose sites were half the bricked volume and half whatever
    the next allocation had written over it -- end-to-end PCC 2%, and invisible to a unit test
    because nothing allocated afterwards. Compare BUFFERS, which is what is being freed.
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


def _reshape_row_major(x: ttnn.Tensor, shape: Sequence[int]) -> ttnn.Tensor:
    """Reshape and leave the result in ROW_MAJOR, for consumers that want it that way.

    Tilizing a shape whose second-to-last dim is small is expensive out of proportion to the
    data: TILE rounds both of the last two dims up to 32, so a trailing ``(num_heads, head_dim)``
    of ``(4, 64)`` costs 8x its own size — 13 GB for a 1.7 GB activation at 1920x1088. The
    neighborhood-attention primitive gathers rows in ROW_MAJOR anyway, so tilizing here only to
    have it undone was paying that padding for nothing.
    """
    return ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), tuple(shape))


@dataclass(frozen=True)
class _Band:
    """A frame band of the volume: interior ``[lo, hi)`` plus the halo attention reaches into.

    ``layout_hi`` is ``hi`` rounded up to a brick-T multiple so a bricked residual stays a
    contiguous ``T_br`` slice. Ghost frames between ``hi`` and ``layout_hi`` are zeros, masked
    by the op (window placement uses the true ``hi``), and cropped at stage exit.
    """

    lo: int
    hi: int
    pad_lo: int
    pad_hi: int
    layout_hi: int

    @property
    def frames(self) -> int:
        return self.hi - self.lo

    @property
    def pad_frames(self) -> int:
        return self.pad_hi - self.pad_lo

    @property
    def layout_frames(self) -> int:
        return self.layout_hi - self.lo


def _align_up(value: int, step: int) -> int:
    return value if step <= 1 else (value + step - 1) // step * step


def _align_down(value: int, step: int) -> int:
    return value if step <= 1 else value - (value % step)


def _bands(t: int, *, frames: int | None, kernel: int, align: int = 1) -> tuple[_Band, ...]:
    """Split ``t`` frames into bands of ``frames``, each with the halo its windows reach into.

    ``frames=None``, or a band long enough to cover everything, gives one band whose halo is
    empty: every slice downstream is then a no-op and the volume runs whole, which is what short
    videos and the parity tests do.

    The halo comes from :func:`window_bounds`, the rule the attention plan itself is built from,
    rather than from half the kernel: a query within half a kernel of either end has its window
    shifted inward instead of truncated, so it reaches as far as ``kernel - 1`` frames the other
    way. Taking the bound from the shared function also means a band's local windows are the global
    ones shifted by ``pad_lo``, so the attention masks a band builds are the volume's own.

    ``align`` rounds lo / pad_lo down and pad_hi / layout_hi up to a multiple of the brick's T
    extent, so those cuts stay contiguous in bricked order. The cheap extra cost is at most
    ``align - 1`` frames of halo per band edge (3 at brick ``(8, 2, 2)`` with window radius 5).
    """
    t_layout = _align_up(t, align)
    if frames is None or frames >= t:
        return (_Band(0, t, 0, t_layout, t_layout),)
    if align > 1:
        frames = max(align, _align_down(frames, align))
        if frames >= t:
            return (_Band(0, t, 0, t_layout, t_layout),)
    starts, ends = window_bounds(t, kernel)
    bands = []
    for lo in range(0, t, frames):
        hi = min(lo + frames, t)
        pad_lo, pad_hi = starts[lo], ends[hi - 1]
        if align > 1:
            pad_lo = _align_down(pad_lo, align)
            pad_hi = min(_align_up(pad_hi, align), t_layout)
        layout_hi = min(_align_up(hi, align), t_layout)
        bands.append(_Band(lo, hi, pad_lo, pad_hi, layout_hi))
    return tuple(bands)


def _slice_rows(x: ttnn.Tensor, lo: int, hi: int) -> ttnn.Tensor:
    """Rows ``[lo, hi)`` of a ``(1, batch, rows, ·)`` tensor, or ``x`` itself if that is every row.

    Returning the input unsliced matters for more than speed: it keeps the single-band path free of
    copies, so running unslabbed costs exactly what it did before bands existed.
    """
    shape = tuple(x.shape)
    if (lo, hi) == (0, shape[-2]):
        return x
    return ttnn.slice(x, [0, 0, lo, 0], [shape[0], shape[1], hi, shape[-1]])


def _slice_last(x: ttnn.Tensor, start: int, stop: int) -> ttnn.Tensor:
    """Slice the channel dim. Callers keep ``start``/``stop`` tile-aligned."""
    starts = [0] * (len(x.shape) - 1) + [start]
    stops = [*list(x.shape)[:-1], stop]
    return ttnn.slice(x, starts, stops)


def _modulate(x: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor) -> ttnn.Tensor:
    """``x * (1 + scale) + shift``. ``scale``/``shift`` broadcast over the site axis."""
    return ttnn.add(ttnn.multiply(x, ttnn.add(scale, 1.0)), shift)


def _modulate_consuming(x: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor) -> ttnn.Tensor:
    """``_modulate`` that frees ``x`` and its own intermediate, for volume-sized inputs."""
    scaled = ttnn.multiply(x, ttnn.add(scale, 1.0))
    ttnn.deallocate(x)
    out = ttnn.add(scaled, shift)
    ttnn.deallocate(scaled)
    return out


def _add_consuming(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    """``a + b``, freeing both operands. Residual adds are where a block's copies pile up."""
    out = ttnn.add(a, b)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return out


def _pad_out_features(w: torch.Tensor, out_features: int) -> torch.Tensor:
    return torch.nn.functional.pad(w, (0, 0, 0, out_features - w.shape[0]))


def _pad_in_features(w: torch.Tensor, in_features: int) -> torch.Tensor:
    return torch.nn.functional.pad(w, (0, in_features - w.shape[1]))


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class _TimestepEmbedder(Module):
    """``PixArtAlphaCombinedTimestepSizeEmbeddings`` with ``size_emb_dim=0``."""

    def __init__(self, t_emb_dim: int, *, mesh_device: ttnn.MeshDevice, dtype: ttnn.DataType) -> None:
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            cos_first=True,
            downscale_freq_shift=0,
            dtype=dtype,
            mesh_device=mesh_device,
        )
        self.mlp = TimestepEmbedding(256, t_emb_dim, act_fn="silu", dtype=dtype, mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # The checkpoint ships a bare Sequential (mlp.0 / mlp.2), not upstream's own
        # timestep_embedder.linear_{1,2} spelling; upstream's loader renames it too.
        for src, dst in (("0", "linear_1"), ("2", "linear_2")):
            for leaf in ("weight", "bias"):
                key = f"mlp.{src}.{leaf}"
                if key in state:
                    state[f"mlp.{dst}.{leaf}"] = state.pop(key)

    def forward(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        return self.mlp(self.time_proj(timestep))


class _NeighborhoodAttention3D(Module):
    """Q/K/V projection, per-head QK RMSNorm, absolute RoPE, NA3D, output projection."""

    def __init__(
        self,
        config: DiffVAEStage5Config,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType,
        ccl_manager=None,
        na3d_backend: str | NAKernel,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        # Which NA3D executor runs: "gather"/"op" run the attention replicated (whole volume on
        # every chip); "op_sp_w_sharded" and "bricked_sp_w_sharded" keep this chip's W-shard of the
        # sequence through the whole attention (K/V reached internally), for full-stage spatial-W SP.
        # Resolved by the stage and handed down, so the three levels cannot pick different backends.
        self.kernel = resolve_na_kernel(na3d_backend)
        self.sp_axis = sp_axis
        # TP-over-heads on a second mesh axis (only meaningful under a w_sharded kernel): the
        # attention runs on heads/tp of the heads per chip, gathered back before the output
        # projection.
        self.tp_axis = tp_axis
        self.scale = config.head_dim**-0.5

        # DIFFVAE_TP_PROJ=1: make the qkv projections COLUMN-PARALLEL over the TP axis (shard the
        # weight on the head-output dim). Each chip then computes only its heads' q/k/v -- a local
        # matmul, no comms -- instead of all heads redundantly, and feeds the attention already
        # head-sharded (its internal head-slice is skipped; the existing head all-gather still runs
        # before the replicated out-proj). qkv is 3 of the 4 projections, so this reclaims the bulk
        # of the ~0.9s redundant projection compute without adding an all-reduce.
        self.tp_proj = tp_axis is not None and os.environ.get("DIFFVAE_TP_PROJ", "1") == "1"
        tp = int(list(mesh_device.shape)[tp_axis]) if self.tp_proj else 1
        assert not self.tp_proj or config.num_heads % tp == 0, f"num_heads={config.num_heads} not divisible by tp={tp}"
        self.heads_local = config.num_heads // tp

        linear = {"bias": True, "mesh_device": mesh_device, "dtype": dtype}
        qkv_linear = dict(linear)
        if self.tp_proj:
            qkv_linear["weight_mesh_axes"] = [None, tp_axis]  # shard the output (heads) over the TP axis
            qkv_linear["bias_mesh_axes"] = [None, tp_axis]
        # DIFFVAE_S5_FUSED_QKV=1: keep the checkpoint's own fused qkv as one matmul instead of
        # splitting it into three. Three column-parallel projections are 256->64 each, a narrow GEMM;
        # fused it is one 256->192. The split then costs three slices of the packed output.
        self.fused_qkv = os.environ.get("DIFFVAE_S5_FUSED_QKV", "0") == "1"
        if self.fused_qkv:
            self.qkv = Linear(config.dim, 3 * config.dim, **qkv_linear)
        else:
            self.to_q = Linear(config.dim, config.dim, **qkv_linear)
            self.to_k = Linear(config.dim, config.dim, **qkv_linear)
            self.to_v = Linear(config.dim, config.dim, **qkv_linear)
        self.proj = Linear(config.dim, config.dim, **linear)  # out-proj stays replicated (full width in)

        norm = {
            "norm_eps": config.norm_eps,
            "bias": False,
            "mesh_device": mesh_device,
            "dtype": dtype,
        }
        self.q_norm = RMSNorm(config.head_dim, **norm)
        self.k_norm = RMSNorm(config.head_dim, **norm)

        self.pair_swap = ttnn.from_torch(
            _rope_pair_swap_matrix(config.head_dim),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
        )
        # The swap matmul only permutes and negates, so it must not lose mantissa bits;
        # ttnn.matmul's default LoFi fidelity would truncate every rotated lane.
        self.swap_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Checkpoints ship one Linear(dim, 3*dim) under qkv.*; the split is [q | k | v]
        # along the output dim.
        if self.fused_qkv:
            # Kept fused, but regrouped: a contiguous column shard must be one device's own heads of
            # all three of q, k and v, where the shipped order would give device 0 nothing but q.
            cfg = self.config
            hd, hl = cfg.head_dim, self.heads_local
            devices = cfg.num_heads // hl
            order = torch.cat(
                [
                    torch.arange(part * cfg.dim + h * hd, part * cfg.dim + (h + 1) * hd)
                    for d in range(devices)
                    for part in range(3)
                    for h in range(d * hl, (d + 1) * hl)
                ]
            )
            for leaf in ("weight", "bias"):
                fused = state.get(f"qkv.{leaf}")
                if fused is not None:
                    state[f"qkv.{leaf}"] = fused[order].clone()
            return
        for leaf in ("weight", "bias"):
            fused = state.pop(f"qkv.{leaf}", None)
            if fused is None:
                continue
            if fused.shape[0] % 3 != 0:
                msg = f"fused qkv.{leaf} leading dim {fused.shape[0]} is not divisible by 3"
                raise ValueError(msg)
            d = fused.shape[0] // 3
            state[f"to_q.{leaf}"] = fused[:d].clone()
            state[f"to_k.{leaf}"] = fused[d : 2 * d].clone()
            state[f"to_v.{leaf}"] = fused[2 * d :].clone()

    def _rope(self, x: ttnn.Tensor, tables: _RopeTables) -> ttnn.Tensor:
        """**Consumes** ``x``, which callers pass as a temporary."""
        return _apply_rope(x, tables, pair_swap=self.pair_swap, compute_kernel_config=self.swap_compute_config)

    def _projected(self, projection, y: ttnn.Tensor, shape: tuple[int, ...]) -> ttnn.Tensor:
        """``projection(y)`` reshaped to per-head form, without keeping the flat result."""
        flat = projection(y)
        out = _reshape_retiled(flat, shape)
        if out is not flat:
            ttnn.deallocate(flat)
        return out

    def _normed(self, norm, x: ttnn.Tensor, *, scale: float | None = None) -> ttnn.Tensor:
        """``norm(x)``, optionally scaled, consuming ``x``."""
        out = norm(x)
        ttnn.deallocate(x)
        if scale is not None:
            scaled = ttnn.multiply(out, scale)
            ttnn.deallocate(out)
            return scaled
        return out

    def forward(
        self, y: ttnn.Tensor, grid: Grid, tables: _RopeTables, brick: tuple[int, int, int] | None = None
    ) -> ttnn.Tensor:
        """``y``: ``(1, batch, sites, dim)``. Returns the same shape.

        ``grid`` is always the FULL ``(T, H, W)``. Under spatial-W SP (``op_sp_w_sharded``) ``y`` is
        this chip's W-shard, so the local W extent is ``W/sp``; the shapes below use that while the
        attention is still told the full W (its executor gathers the missing columns). ``tables``
        must be W-sharded to match ``y`` in that mode (frame piece over this chip's H×(W/sp) rows).

        ``brick`` set means ``y`` is already in bricked site order (stage-5 hoist): Q/K/V stay
        bricked, RoPE uses the fused bricked table, and the op is told ``already_bricked``.
        """
        cfg = self.config
        assert grid.batch == 1, f"batched stage 5 is not implemented; got batch={grid.batch}"
        sharded = self.kernel.w_sharded
        if sharded:
            sp = int(list(self.mesh_device.shape)[self.sp_axis])
            assert grid.w % sp == 0, f"W={grid.w} must split evenly over sp={sp}"
            w_local = grid.w // sp
        else:
            w_local = grid.w
        sites_local = (
            brick_count((grid.t, grid.h, w_local), brick) * SITES_PER_BRICK
            if brick is not None
            else grid.t * grid.h * w_local
        )
        # Frames are a separate axis rather than folded into the rows, which is what lets the RoPE
        # pieces broadcast: the H/W piece over frames, the T piece over the rows within one. Under
        # column-parallel qkv (tp_proj) the projections already emit only this chip's heads, so the
        # per-head shapes use the local head count and the attention is told the heads are presharded.
        # Bricked RoPE is fused -- one row per (site, head) -- so the T axis collapses.
        heads = self.heads_local
        heads_shape = (
            (1, 1, sites_local * heads, cfg.head_dim)
            if brick is not None
            else (1, grid.t, grid.h * w_local * heads, cfg.head_dim)
        )
        volume_shape = (grid.batch, grid.t, grid.h, w_local, heads, cfg.head_dim)

        def to_volume(x: ttnn.Tensor) -> ttnn.Tensor:
            """Untilize into the volume shape NA3D gathers from, consuming ``x``."""
            out = _reshape_row_major(x, volume_shape)
            if out is not x:
                ttnn.deallocate(x)
            return out

        def to_flat(x: ttnn.Tensor) -> ttnn.Tensor:
            """Merge the frame axis into the rows, giving the (B, NH, S, HD) the flat path wants.

            A view rather than a retile: rows per frame is ``H*(W/sp)``, a whole number of tiles, so
            the tile grid only grows taller and no tile is re-cut.
            """
            return ttnn.reshape(x, (grid.batch, heads, sites_local, cfg.head_dim))

        # The flat (B, NH, S, HD) sequence, where the kernel wants it AND the shape allows it. One
        # head per chip is the hard part: above that a frame's rows are site-major with heads inner,
        # so the flat form would want a real permute rather than a frame-axis merge -- which is only
        # reachable under column-parallel qkv with heads == tp, i.e. TP_HEADS=1. Both preconditions
        # are runtime facts about this call rather than properties of the kernel, so they stay here;
        # the executor asserts heads_presharded for the flat path too.
        flat_seq = self.kernel.prefers_flat_seq and self.tp_proj and heads == 1
        if brick is not None:
            # Already bricked: RoPE output is (1, 1, sites*heads, hd); fold back to (B, heads, sites, hd).
            # A RELABEL, not a copy -- the executor reads the same site-major buffer back out as
            # (B, 1, sites, heads*hd). So nothing is freed here: ttnn.reshape hands back a new
            # wrapper over the SAME buffer, and deallocating the input takes the reshape's memory
            # with it (under TP the shapes already match, where it took the tensor's device too and
            # crashed outright; above one head it silently returned half-overwritten activations).
            def to_bricked_seq(x: ttnn.Tensor) -> ttnn.Tensor:
                target = (grid.batch, heads, sites_local, cfg.head_dim)
                if tuple(x.shape) == target:
                    return x
                out = ttnn.reshape(x, target)
                _release_intermediates((x,), keep=out)
                return out

            prep = to_bricked_seq
        else:
            prep = to_flat if flat_seq else to_volume

        # Built and consumed one at a time. Holding q, k and v plus each one's untilized copy
        # and RoPE temporaries is what exhausts DRAM at full resolution -- which is also why the
        # fused path slices its packed output a lane at a time rather than all three up front.
        if self.fused_qkv:
            with deep_prof(self.mesh_device, "qkv-proj", category=decode_tree.PROJ):
                packed = self.qkv(y)
            width = self.heads_local * cfg.head_dim

            def lane(index: int) -> ttnn.Tensor:
                part = _slice_last(packed, index * width, (index + 1) * width)
                out = _reshape_retiled(part, heads_shape)
                if out is not part:
                    ttnn.deallocate(part)
                return out

            with deep_prof(self.mesh_device, "qkv-lanes: slice+norm+rope", category=decode_tree.NORM_ROPE):
                q = prep(self._rope(self._normed(self.q_norm, lane(0), scale=self.scale), tables))
                k = prep(self._rope(self._normed(self.k_norm, lane(1)), tables))
                v = prep(lane(2))
                ttnn.deallocate(packed)
        else:
            # Still one lane at a time (see above): the spans name each step without hoisting any
            # projection past the norm/rope that consumes it, so peak DRAM is unchanged. All three
            # lanes share a label, so the tree pools them into one row per step with n=3.
            def lane_unfused(projection, norm, *, scale=None, rope=True):
                with deep_prof(self.mesh_device, "qkv-proj", category=decode_tree.PROJ):
                    part = self._projected(projection, y, heads_shape)
                if norm is not None:
                    with deep_prof(self.mesh_device, "qkv-norm", category=decode_tree.NORM_ROPE):
                        part = self._normed(norm, part, scale=scale)
                    if rope:
                        with deep_prof(self.mesh_device, "qkv-rope", category=decode_tree.NORM_ROPE):
                            part = self._rope(part, tables)
                with deep_prof(self.mesh_device, "qkv-prep (to seq/volume)", category=decode_tree.RESHAPE):
                    return prep(part)

            q = lane_unfused(self.to_q, self.q_norm, scale=self.scale)
            k = lane_unfused(self.to_k, self.k_norm)
            v = lane_unfused(self.to_v, None)

        # The one place the executor is chosen. Everything that used to guard these arms -- the
        # W-shard membership test, the flat-sequence opt-in, its per-backend override -- is now a
        # field of the record being matched, so no two of them can disagree.
        match self.kernel.name:
            case "bricked_sp_w_sharded":
                # Our op: this chip's W-shard plus the halo its windows reach into, and a per-device
                # gather table carrying where that shard sits in the global volume. Window placement
                # stays global, so a query near a shard seam still sees a full window.
                from ...layers.neighborhood_attention import neighborhood_attention_3d_bricked_w_sharded

                out = neighborhood_attention_3d_bricked_w_sharded(
                    q,
                    k,
                    v,
                    dims=(grid.t, grid.h, grid.w),
                    kernel_size=cfg.kernel_size,
                    sp_axis=self.sp_axis,
                    ccl_manager=self.ccl_manager,
                    scale=1.0,
                    tp_axis=self.tp_axis,
                    heads_presharded=self.tp_proj,
                    already_bricked=brick is not None,
                    brick=brick,
                )
            case "op_sp_w_sharded":
                out = neighborhood_attention_3d_op_sp_w_sharded(
                    q,
                    k,
                    v,
                    dims=(grid.t, grid.h, grid.w),
                    kernel_size=cfg.kernel_size,
                    sp_axis=self.sp_axis,
                    ccl_manager=self.ccl_manager,
                    scale=1.0,
                    tp_axis=self.tp_axis,
                    heads_presharded=self.tp_proj,
                    flat_seq=flat_seq,
                    gna_stride=None if cfg.gna_stride == (1, 1, 1) else cfg.gna_stride,
                )
            case _:
                # Replicated: the whole volume on every chip. na3d's own dispatcher picks the
                # executor from the same name.
                out = neighborhood_attention_3d(
                    q,
                    k,
                    v,
                    kernel_size=cfg.kernel_size,
                    scale=1.0,
                    ccl_manager=self.ccl_manager,
                    backend=self.kernel.name,
                )
        for tensor in (q, k, v):
            ttnn.deallocate(tensor)

        with deep_prof(self.mesh_device, "out-proj", category=decode_tree.PROJ):
            flat = _reshape_retiled(out, (1, grid.batch, sites_local, cfg.dim))
            if flat is not out:
                ttnn.deallocate(out)
            projected = self.proj(flat)
            ttnn.deallocate(flat)
        return projected


class DiffusionNABlock(Module):
    """Context injection, then AdaLN residual attention, then AdaLN residual SwiGLU."""

    def __init__(
        self,
        config: DiffVAEStage5Config,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType,
        ccl_manager=None,
        na3d_backend: str | NAKernel,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        # Resolve the backend once so the block and its attention agree on whether the sequence is
        # W-sharded: under a w_sharded kernel the per-chip tensor holds only H*(W/sp) rows per frame,
        # so the block's own frame slicing must use the local rows-per-frame, not the full W.
        self.kernel = resolve_na_kernel(na3d_backend)
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.context_proj = Linear(config.context_channels, config.dim, bias=True, mesh_device=mesh_device, dtype=dtype)
        self.scale_shift_table = Parameter(
            total_shape=[1, NUM_ADALN_CHUNKS * config.dim], device=mesh_device, dtype=dtype
        )

        norm = {
            "norm_eps": config.norm_eps,
            "bias": False,
            "mesh_device": mesh_device,
            "dtype": dtype,
        }
        self.norm1 = RMSNorm(config.dim, **norm)
        self.attn = _NeighborhoodAttention3D(
            config,
            mesh_device=mesh_device,
            dtype=dtype,
            ccl_manager=ccl_manager,
            na3d_backend=self.kernel,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )
        self.norm2 = RMSNorm(config.dim, **norm)
        # Fused [up | gate] projection: Linear's swiglu path packs the two halves into
        # one GEMM and emits silu(gate) * up.
        self.mlp_gate_up = Linear(
            config.dim, config.mlp_hidden, bias=False, activation_fn="swiglu", mesh_device=mesh_device, dtype=dtype
        )
        self.mlp_down = Linear(config.mlp_hidden, config.dim, bias=False, mesh_device=mesh_device, dtype=dtype)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "scale_shift_table" in state:
            state["scale_shift_table"] = state["scale_shift_table"].reshape(1, -1)

        gate = state.pop("mlp.w_gate.weight", None)
        up = state.pop("mlp.w_up.weight", None)
        if gate is not None and up is not None:
            state["mlp_gate_up.weight"] = torch.cat([up, gate], dim=0)
        down = state.pop("mlp.w_down.weight", None)
        if down is not None:
            state["mlp_down.weight"] = down

    def forward(
        self,
        x: list[ttnn.Tensor],
        context: ttnn.Tensor,
        shared_modulation: ttnn.Tensor,
        grid: Grid,
        bands: tuple[_Band, ...],
        tables: tuple[_RopeTables, ...],
        brick: tuple[int, int, int] | None = None,
    ) -> list[ttnn.Tensor]:
        """``x``: the volume as one ``(1, batch, band rows, dim)`` tensor per band, and the return
        is the updated volume in the same form. **Consumes** ``x``; ``context`` is read by every
        block and left alone.

        Attention is the only part that reads across sites, so each band runs on its own rows and
        only the attention sees the band plus its halo, cropped back afterwards. Because the halo is
        exactly what :func:`window_bounds` reaches (see :func:`_bands`), a padded band's local
        windows are the volume's own shifted by ``pad_lo``, so every query kept sees the window it
        would have seen whole and the crop is exact. One band is the whole volume with an empty
        halo, so that path is unchanged.

        Written as a sequence with each temporary freed as it dies rather than as nested
        expressions, which is worth the verbosity: the nested form holds five or six band-sized
        tensors at once and that is most of a block's peak. ``mod`` and its chunks are
        ``(1, batch, 1, ·)``, so they cost nothing.
        """
        dim = self.config.dim
        # Adding the table to all 7 chunks and then reading 4 is upstream's
        # _modulation: the gate chunks are computed and discarded.
        mod = ttnn.add(shared_modulation, self.scale_shift_table.data)
        scale_msa = _slice_last(mod, 0, dim)
        shift_msa = _slice_last(mod, dim, 2 * dim)
        scale_mlp = _slice_last(mod, 3 * dim, 4 * dim)
        shift_mlp = _slice_last(mod, 4 * dim, 5 * dim)

        # Rows per frame on THIS chip. Under spatial-W SP the sequence is W-sharded, so a frame holds
        # only H*(W/sp) rows here; the frame-granular band slicing below must use that local count.
        # Bricked order groups sites by T_br, so the slice unit is one T-brick of sites instead.
        if self.kernel.w_sharded:
            sp = int(list(self.mesh_device.shape)[self.sp_axis])
            w_local = grid.w // sp
            rows = sites_per_t_brick((grid.t, grid.h, w_local), brick) if brick is not None else grid.h * w_local
        else:
            rows = sites_per_t_brick((grid.t, grid.h, grid.w), brick) if brick is not None else grid.h * grid.w
        frame_step = brick[0] if brick is not None else 1
        # A local view of the volume so the caller's list is left alone; entries become None as
        # this loop releases them.
        live: list[ttnn.Tensor | None] = list(x)
        out: list[ttnn.Tensor] = []
        for index, band in enumerate(bands):
            # Bands own nothing they were handed: ``live`` is freed by this loop's own bookkeeping
            # below, so anything derived from it is released here the moment it stops being read.
            with deep_prof(self.mesh_device, "halo assemble (padded rows)", category=decode_tree.RESHAPE):
                padded = self._padded_rows(live, index, bands, rows, frame_step=frame_step)
            interior = (
                (band.lo - band.pad_lo) // frame_step * rows,
                (band.layout_hi - band.pad_lo) // frame_step * rows,
            )

            with block_prof(self.mesh_device, "context-inject", category=decode_tree.CONTEXT_INJECT):
                context_rows = _slice_rows(context, band.pad_lo // frame_step * rows, band.pad_hi // frame_step * rows)
                injected = self.context_proj(context_rows)
                if context_rows is not context:
                    ttnn.deallocate(context_rows)
                xs = ttnn.add(padded, injected)
                ttnn.deallocate(injected)
                if padded is not live[index]:
                    ttnn.deallocate(padded)

            with deep_prof(self.mesh_device, "norm+modulate (pre-attn)", category=decode_tree.NORM_ROPE):
                modulated = _modulate_consuming(self.norm1(xs), scale_msa, shift_msa)
            with block_prof(self.mesh_device, "attention", category=decode_tree.ATTENTION):
                attended = self.attn(
                    modulated,
                    Grid(grid.batch, min(band.pad_hi, grid.t) - band.pad_lo, grid.h, grid.w),
                    tables[index],
                    brick=brick,
                )
            ttnn.deallocate(modulated)

            with deep_prof(self.mesh_device, "residual crop+add (attn)", category=decode_tree.RESHAPE):
                residual = _slice_rows(xs, *interior)
                if residual is not xs:
                    ttnn.deallocate(xs)
                cropped = _slice_rows(attended, *interior)
                if cropped is not attended:
                    ttnn.deallocate(attended)
                y = _add_consuming(residual, cropped)

            with deep_prof(self.mesh_device, "norm+modulate (pre-mlp)", category=decode_tree.NORM_ROPE):
                modulated = _modulate_consuming(self.norm2(y), scale_mlp, shift_mlp)
            with block_prof(self.mesh_device, "mlp", category=decode_tree.MLP):
                hidden = self.mlp_gate_up(modulated)
                ttnn.deallocate(modulated)
                projected = self.mlp_down(hidden)
                ttnn.deallocate(hidden)
                out.append(_add_consuming(y, projected))

            # A band's input rows are read as halo by its neighbours, so they die a beat after the
            # band itself. Releasing them as soon as no band still to come reaches back that far is
            # what keeps the volume from being resident twice while the new one is built up.
            reach = bands[index + 1].pad_lo if index + 1 < len(bands) else bands[-1].pad_hi
            for other, entry in enumerate(live):
                if entry is not None and bands[other].hi <= reach:
                    ttnn.deallocate(entry)
                    live[other] = None
        return out

    def _padded_rows(
        self,
        live: list[ttnn.Tensor | None],
        index: int,
        bands: tuple[_Band, ...],
        rows: int,
        frame_step: int = 1,
    ) -> ttnn.Tensor:
        """Band ``index``'s rows plus its halo, read out of whichever bands the halo spans."""
        band = bands[index]
        parts = []
        for other, source in enumerate(bands):
            lo = max(band.pad_lo, source.lo)
            hi = min(band.pad_hi, source.layout_hi)
            if lo < hi:
                assert live[other] is not None, f"band {other} was released before band {index} read it"
                parts.append(
                    _slice_rows(
                        live[other],
                        (lo - source.lo) // frame_step * rows,
                        (hi - source.lo) // frame_step * rows,
                    )
                )
        if len(parts) == 1:
            return parts[0]
        joined = ttnn.concat(parts, dim=-2)
        # The halo slices are copies, but a part that happens to be a whole band is that band's own
        # tensor, borrowed: the bands around this one still have to read it.
        for part in parts:
            if not any(part is entry for entry in live):
                ttnn.deallocate(part)
        return joined


class _SharedAdaLNZero(Module):
    """``proj(silu(t_emb))`` broadcast to the per-site modulation layout."""

    def __init__(
        self,
        config: DiffVAEStage5Config,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType,
        out_dtype: ttnn.DataType,
    ) -> None:
        super().__init__()
        self.out_dtype = out_dtype
        self.proj = Linear(
            config.t_emb_dim,
            NUM_ADALN_CHUNKS * config.dim,
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def forward(self, t_emb: ttnn.Tensor, batch: int) -> ttnn.Tensor:
        """``t_emb``: ``(1, 1, batch, t_emb_dim)``. Returns ``(1, batch, 1, 7 * dim)``.

        Batch moves to dim 1 so a single size-1 site axis broadcasts the chunks over the
        whole volume, which is upstream's ``(B, 1, 1, 1, C)`` view.
        """
        mod = self.proj(ttnn.silu(t_emb))
        if mod.dtype != self.out_dtype:
            mod = ttnn.typecast(mod, self.out_dtype)
        return _reshape_retiled(mod, (1, batch, 1, mod.shape[-1]))


class DiffVAEStage5(Module):
    """The stage-5 diffusion stack: ``forward_diff_step`` plus patch packing.

    ``patchify``/``unpatchify`` run on the host: both are 8-axis permutations in
    channels-last pixel space that ttnn cannot express, and they sit at the module
    boundary where a single transfer is cheap.
    """

    def __init__(
        self,
        config: DiffVAEStage5Config | None = None,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        modulation_dtype: ttnn.DataType = ttnn.float32,
        ccl_manager=None,
        na3d_backend: str | NAKernel | None = None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DiffVAEStage5Config()
        cfg = self.config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.dtype = dtype
        self.modulation_dtype = modulation_dtype
        # Spatial-W SP: under a w_sharded kernel the whole stage keeps its sequence W-sharded --
        # context and x_t are uploaded/resharded over W, the blocks run 1/sp, and the tail output is
        # gathered back over W in forward. This is the ONE place the backend is resolved; the blocks
        # and their attention are handed the record, so no level can pick a different one.
        self.kernel = resolve_na_kernel(na3d_backend or "gather")
        self.sp_axis = sp_axis
        # TP-over-heads on a second mesh axis: only the per-head attention shards over it; every
        # other op (context, RoPE, MLP, tail) stays replicated across tp_axis. Composes with the
        # W-shard above -- the two use orthogonal mesh axes.
        self.tp_axis = tp_axis
        self._w_sharded = self.kernel.w_sharded
        # Hoist brick conversion to stage entry/exit rather than paying it per block: that 7-D
        # permute was 735 ms of the decode on the one path that needs it.
        self._keep_bricked = self.kernel.keep_bricked
        self._brick: tuple[int, int, int] | None = None
        # Under column-parallel qkv (DIFFVAE_TP_PROJ) the q/k carry only heads/tp per chip, so the
        # shared RoPE tables (which repeat each row per head) must be built for the local head count.
        _tp_proj = tp_axis is not None and os.environ.get("DIFFVAE_TP_PROJ", "1") == "1"
        _tp = int(list(mesh_device.shape)[tp_axis]) if _tp_proj else 1
        self._rope_num_heads = self.config.num_heads // _tp
        if self._w_sharded:
            assert sp_axis is not None, f"{self.kernel.name} needs sp_axis"
            assert ccl_manager is not None, f"{self.kernel.name} needs a ccl_manager"
        # The tile-aligned width the 48 patch channels are zero-padded to. The pad has to
        # be explicit on conv_in_x_t's K axis: a garbage-filled activation tail would
        # otherwise multiply against whatever the weight's own tile pad happens to hold.
        self.padded_patch_channels = math.ceil(cfg.patch_channels / TILE) * TILE
        self._rope_cache: dict[Grid, _RopeTables] = {}

        self.conv_in_x_t = Linear(self.padded_patch_channels, cfg.dim, bias=True, mesh_device=mesh_device, dtype=dtype)
        # The timestep sinusoid and everything downstream of it feed multiplicative
        # modulation in every block, so keep that short chain at higher precision.
        self.t_embedder = _TimestepEmbedder(cfg.t_emb_dim, mesh_device=mesh_device, dtype=modulation_dtype)
        self.shared_adaln = _SharedAdaLNZero(cfg, mesh_device=mesh_device, dtype=modulation_dtype, out_dtype=dtype)
        self.diff_blocks = ModuleList(
            DiffusionNABlock(
                cfg,
                mesh_device=mesh_device,
                dtype=dtype,
                ccl_manager=ccl_manager,
                na3d_backend=self.kernel,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
            )
            for _ in range(cfg.num_blocks)
        )
        self.norm_out = RMSNorm(cfg.dim, norm_eps=cfg.norm_eps, bias=False, mesh_device=mesh_device, dtype=dtype)
        self.conv_out = Linear(cfg.dim, self.padded_patch_channels, bias=True, mesh_device=mesh_device, dtype=dtype)

    GATE_SUFFIXES = (".gate_msa", ".gate_mlp", ".gate_ctx")

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Static gates were folded into attn.proj / mlp.w_down / context_proj at export.
        # An unfolded checkpoint would silently decode wrong, so refuse it rather than
        # trusting the shipped variant. Checked at the root so one pass covers every block.
        gates = sorted(k for k in state if k.endswith(self.GATE_SUFFIXES))
        if gates:
            msg = f"unfolded static gates present ({', '.join(gates)}); this port assumes pre-folded weights"
            raise ValueError(msg)

        if "conv_in_x_t.weight" in state:
            state["conv_in_x_t.weight"] = _pad_in_features(state["conv_in_x_t.weight"], self.padded_patch_channels)
        if "conv_out.weight" in state:
            state["conv_out.weight"] = _pad_out_features(state["conv_out.weight"], self.padded_patch_channels)
        if "conv_out.bias" in state:
            bias = state["conv_out.bias"]
            state["conv_out.bias"] = torch.nn.functional.pad(bias, (0, self.padded_patch_channels - bias.shape[0]))

    def _stage5_brick(self, grid: Grid) -> tuple[int, int, int]:
        """The brick the whole stage converts with -- same choice the attention op would make."""
        if self._brick is not None:
            return self._brick
        from ...layers.neighborhood_attention import _choose_sharded_brick, configured_stride

        sp = int(list(self.mesh_device.shape)[self.sp_axis]) if self._w_sharded else 1
        w_local = grid.w // sp
        volume = (grid.t, grid.h, grid.w)
        context_window = tuple(min(window, extent) for window, extent in zip(self.config.kernel_size, volume))
        brick_env = os.environ.get("DIFFVAE_NA_BRICK")
        self._brick = (
            tuple(int(part) for part in brick_env.split(","))
            if brick_env
            else _choose_sharded_brick(volume, context_window, configured_stride(), w_local, sp)
        )
        return self._brick

    def _local_volume(self, grid: Grid, t: int | None = None) -> tuple[int, int, int]:
        sp = int(list(self.mesh_device.shape)[self.sp_axis]) if self._w_sharded else 1
        return (grid.t if t is None else t, grid.h, grid.w // sp)

    def _brick_activation(
        self, x: ttnn.Tensor, volume: tuple[int, int, int], brick: tuple[int, int, int]
    ) -> ttnn.Tensor:
        """``(1, batch, T*H*W, C)`` TILE natural -> ``(1, batch, bricked_sites, C)`` TILE."""
        channels = int(x.shape[-1])
        batch = int(x.shape[1])
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        vol = ttnn.reshape(rm, (batch, volume[0], volume[1], volume[2], channels))
        grid5 = to_bricked_grid(vol, volume=volume, brick=brick)
        flat = ttnn.reshape(grid5, (1, batch, brick_count(volume, brick) * SITES_PER_BRICK, channels))
        out = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
        # Freed together, at the end, by buffer rather than by object: vol is a view of rm and
        # flat is one of grid5, so freeing them as they are superseded frees memory that the next
        # step still reads. See _release_intermediates.
        _release_intermediates((x, rm, vol, grid5, flat), keep=out)
        return out

    def _unbrick_activation(
        self, x: ttnn.Tensor, volume: tuple[int, int, int], brick: tuple[int, int, int]
    ) -> ttnn.Tensor:
        """Inverse of :meth:`_brick_activation`, ghosts cropped."""
        channels = int(x.shape[-1])
        batch = int(x.shape[1])
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        bricks_t, bricks_h, bricks_w = brick_grid(volume, brick)
        grid5 = ttnn.reshape(rm, (batch, bricks_t, bricks_h, bricks_w, SITES_PER_BRICK * channels))
        natural = from_bricked_grid(grid5, volume=volume, brick=brick)
        flat = ttnn.reshape(natural, (1, batch, volume[0] * volume[1] * volume[2], channels))
        return ttnn.to_layout(flat, ttnn.TILE_LAYOUT)

    def rope_tables(self, grid: Grid) -> _RopeTables:
        brick = self._stage5_brick(grid) if self._keep_bricked else None
        key = (grid, brick)
        tables = self._rope_cache.get(key)
        if tables is None:
            w_shard = (int(list(self.mesh_device.shape)[self.sp_axis]), self.sp_axis) if self._w_sharded else None
            if brick is not None:
                tables = _build_bricked_rope_tables(
                    grid,
                    brick,
                    dim_split=self.config.resolved_rope_dim_split,
                    base=self.config.rope_base,
                    num_heads=self._rope_num_heads,
                    mesh_device=self.mesh_device,
                    dtype=self.dtype,
                    w_shard=w_shard,
                )
            else:
                tables = _build_rope_tables(
                    grid,
                    dim_split=self.config.resolved_rope_dim_split,
                    base=self.config.rope_base,
                    num_heads=self._rope_num_heads,
                    mesh_device=self.mesh_device,
                    dtype=self.dtype,
                    w_shard=w_shard,
                )
            self._rope_cache[key] = tables
        return tables

    def bands(self, grid: Grid) -> tuple[_Band, ...]:
        """How to split the volume into frame bands, from ``DIFFVAE_SLAB_FRAMES``.

        Off by default: banding buys peak memory with redundant halo work, so it is not worth
        paying until the volume needs it. The row slices are on a tiled dim, so a band boundary has
        to be tile-aligned, which holds exactly when ``h * w`` is a multiple of ``TILE``.
        """
        frames = os.environ.get("DIFFVAE_SLAB_FRAMES")
        kernel = self.config.kernel_size[0]
        if frames and (grid.h * grid.w) % TILE != 0:
            logger.warning(
                f"[diffvae] ignoring DIFFVAE_SLAB_FRAMES: h*w={grid.h * grid.w} is not a multiple of {TILE}, "
                "so a frame boundary is not a tile boundary"
            )
            frames = None
        align = self._stage5_brick(grid)[0] if self._keep_bricked else 1
        return _bands(grid.t, frames=int(frames) if frames else None, kernel=kernel, align=align)

    def device_x_t(self, grid: Grid, bands: tuple[_Band, ...], *, seed: int = 0) -> list[ttnn.Tensor]:
        """x_t noise drawn on device, already in the patchified layout. One tensor per band.

        Patchify is a permutation of iid samples, so drawing at the destination layout is the same
        distribution as drawing pixel-space noise and reshuffling it -- which is what lets this skip a
        908M-element host randn plus the patchify, channel pad, W-reorder and upload. For the same
        reason the shard needs no W-band reorder: any contiguous slice of iid draws is iid, and the
        model only requires that each site's noise be an independent sample, not that it came from a
        particular draw.

        Channels past ``patch_channels`` land on the zero-padded columns of ``conv_in_x_t``, so filling
        them is inert. Drawn at the FULL volume and partitioned rather than per-chip: ttnn.randn
        replicates across the mesh, so a per-chip draw hands every W-band identical values.
        """
        sp = int(list(self.mesh_device.shape)[self.sp_axis]) if self._w_sharded else 1
        out = []
        for index, band in enumerate(bands):
            rows = (band.hi - band.lo) * grid.h * grid.w
            assert rows % sp == 0, f"band rows {rows} not divisible by sp={sp}"
            full = ttnn.randn(
                [1, grid.batch, rows, self.padded_patch_channels],
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                seed=seed + index,
            )
            if sp > 1:
                local = ttnn.mesh_partition(full, dim=2, cluster_axis=self.sp_axis)
                ttnn.deallocate(full)
                full = local
            out.append(self.conv_in_x_t(full))
            ttnn.deallocate(full)
        return out

    def embed_x_t(self, x_t: torch.Tensor, bands: tuple[_Band, ...]) -> list[ttnn.Tensor]:
        """Patchify pixel-space ``(B, C, T, H, W)`` noise and project it, one tensor per band.

        Uploaded a band at a time rather than whole and split afterwards, which would need the
        volume and its projection resident together.
        """
        cfg = self.config
        patched = patchify(x_t, cfg.patch_size)
        batch = patched.shape[0]
        sp = int(list(self.mesh_device.shape)[self.sp_axis]) if self._w_sharded else 1
        out = []
        for band in bands:
            rows = patched[:, :, band.lo : band.hi]
            t, h, w = rows.shape[2:]
            flat = rows.permute(0, 2, 3, 4, 1).reshape(1, batch, t * h * w, cfg.patch_channels)
            flat = torch.nn.functional.pad(flat, (0, self.padded_patch_channels - cfg.patch_channels))
            if self._w_sharded:
                # Reorder the (t, h, w) rows to (device, t, h, w_local) contiguous so from_torch hands
                # device p its W-band, then upload sharded on the site dim -- the sequence never lands
                # whole on any chip.
                w_local = w // sp
                reordered = (
                    flat.reshape(1, batch, t, h, sp, w_local, self.padded_patch_channels)
                    .permute(0, 1, 4, 2, 3, 5, 6)
                    .reshape(1, batch, sp * t * h * w_local, self.padded_patch_channels)
                )
                uploaded = sharded_from_torch(
                    reordered.contiguous(),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=self.dtype,
                    mesh_axes=[None, None, self.sp_axis, None],
                )
            else:
                uploaded = ttnn.from_torch(
                    flat.contiguous(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=self.dtype
                )
            out.append(self.conv_in_x_t(uploaded))
            ttnn.deallocate(uploaded)
        return out

    def forward_diff_step(
        self,
        context: ttnn.Tensor,
        x: list[ttnn.Tensor],
        timestep: ttnn.Tensor,
        grid: Grid,
        bands: tuple[_Band, ...],
        brick: tuple[int, int, int] | None = None,
    ) -> ttnn.Tensor:
        """One stage-5 step. Returns padded patch channels at ``(1, batch, sites, ·)``.

        Upstream carries context and x as one ``[context | conv_in_x_t(x)]`` buffer of
        ``2 * dim`` channels and slices it here. They are kept apart instead: both halves are
        tile-aligned, so building the joint buffer and splitting it straight back is an exact
        round trip, and at 1920x1088 it costs 3.3 GB plus two full-size copies for nothing. No
        block writes the context half, so nothing depends on them being adjacent.
        """
        cfg = self.config
        with stage_timer(self.mesh_device, "stage5 setup: AdaLN + rope tables", category=decode_tree.SETUP):
            scaled_t = ttnn.multiply(timestep, cfg.timestep_scale_multiplier)
            modulation = self.shared_adaln(self.t_embedder(scaled_t), grid.batch)
            tables = self.rope_tables(grid)
            band_tables = tuple(tables.frames(band.pad_lo, band.pad_hi) for band in bands)
        log_dram(self.mesh_device, f"stage5 entry ({len(bands)} band(s))")
        from ...layers.na3d import SP_W_PROF

        _BLOCK_PROF.clear()
        SP_W_PROF.clear()
        for index, block in enumerate(self.diff_blocks):
            with stage_timer(self.mesh_device, f"  stage5 block {index}"):
                x = block(x, context, modulation, grid, bands, band_tables, brick=brick)
            log_dram(self.mesh_device, f"stage5 block {index}")
        # The tail runs per band too: its output is a quarter the width of the volume it comes
        # from, so joining after the projection rather than before is the cheap order.
        tail = []
        for tensor, band in zip(x, bands):
            projected = self.conv_out(self.norm_out(tensor))
            ttnn.deallocate(tensor)
            if brick is not None:
                projected = self._unbrick_activation(projected, self._local_volume(grid, t=band.hi - band.lo), brick)
            tail.append(projected)
        if len(tail) == 1:
            return tail[0]
        joined = ttnn.concat(tail, dim=-2)
        for part in tail:
            ttnn.deallocate(part)
        return joined

    def _wshard_context(self, context: ttnn.Tensor, grid: Grid) -> ttnn.Tensor:
        """Reshard a replicated ``(1, batch, T*H*W, dim)`` context into this chip's W-band.

        The deterministic stages hand context over replicated (T-outer flat), so it is resharded on
        device: a contiguous W-band is not a contiguous slice of the T-outer sequence, so it is
        reshaped to a ``(1, T, H, W, dim)`` volume, ``mesh_partition``ed on W, and flattened back to
        this chip's ``(1, batch, T*H*(W/sp), dim)`` in the same (t, h, w_local) order the blocks use.

        **Consumes** ``context``: the full replicated volume is freed here, since holding it through
        the blocks (9.7 GiB at 6s) is what leaves no room for the per-block K/V gather. This is a
        replicated -> W-sharded reshard; det-stage SP would remove it by handing context W-sharded.
        """
        dim = int(context.shape[-1])
        rm = ttnn.to_layout(context, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(context)
        vol = ttnn.reshape(rm, (1, grid.t, grid.h, grid.w, dim))
        band = ttnn.mesh_partition(vol, dim=3, cluster_axis=self.sp_axis)  # (1, T, H, W/sp, dim)
        ttnn.deallocate(rm)
        w_local = grid.w // int(list(self.mesh_device.shape)[self.sp_axis])
        flat = ttnn.reshape(band, (1, grid.batch, grid.t * grid.h * w_local, dim))
        return ttnn.to_layout(flat, ttnn.TILE_LAYOUT)

    def forward(
        self,
        context: ttnn.Tensor,
        x_t: torch.Tensor | None,
        timestep: ttnn.Tensor,
        grid: Grid,
        *,
        context_sharded: bool = False,
        seed: int = 0,
        device_out: bool = False,
        output_type: str = "float",
    ) -> torch.Tensor | ttnn.Tensor:
        """Return pixels. Valid as the whole decode only for ``model_output_type="x0"``
        with a single inference step, which is what the shipped 2.5 DiffVAE config asks for.

        ``device_out=True`` returns pixels still on device, stopping immediately before the PCIe
        pull -- the channel trim and the depth-to-space unpatchify are device ops and stay inside.
        A trace region cannot contain the pull, so a caller capturing the decode takes this and
        transfers it itself.

        ``context_sharded=True`` means the det stages already handed the context over W-sharded (this
        chip's ``(1, 1, T*H*(W/sp), dim)`` band, same ``sp_axis``), so the re-shard is skipped.
        """
        cfg = self.config
        bands = self.bands(grid)
        with stage_timer(self.mesh_device, "stage5: context reshard", category=decode_tree.RESHAPE):
            if self._w_sharded and not context_sharded:
                context = self._wshard_context(context, grid)
            elif self._w_sharded:
                context = ttnn.to_layout(context, ttnn.TILE_LAYOUT)  # already this chip's band; just ensure TILE

        # Evaluated apart from the block timer: as an argument it was host patchify and a noise
        # upload being charged to the diffusion blocks.
        # Which of the two this is, is known before the clock starts, so the span can name itself at
        # open rather than picking a label on the way out.
        _label = "stage5: device randn + embed x_t" if x_t is None else "stage5: host patchify + embed x_t"
        with stage_timer(self.mesh_device, _label, category=decode_tree.HOST_COMPUTE):
            x_bands = self.device_x_t(grid, bands, seed=seed) if x_t is None else self.embed_x_t(x_t, bands)

        brick = self._stage5_brick(grid) if self._keep_bricked else None
        if brick is not None:
            with stage_timer(self.mesh_device, "stage5: brick x+context", category=decode_tree.RESHAPE):
                context = self._brick_activation(context, self._local_volume(grid), brick)
                x_bands = [
                    self._brick_activation(band_x, self._local_volume(grid, t=band.hi - band.lo), brick)
                    for band_x, band in zip(x_bands, bands)
                ]

        with stage_timer(self.mesh_device, "stage5 diff-blocks (attn+MLP)"):
            out = self.forward_diff_step(context, x_bands, timestep, grid, bands, brick=brick)

        return self._to_pixels(out, grid, device_out=device_out, output_type=output_type)

    def _to_pixels(self, out, grid, *, device_out: bool = False, output_type: str = "float"):
        # The tail splits into a device->host PCIe pull and a host-side unpatchify permute; timing
        # them apart tells us which one the (large, at 1080p) tail cost actually is.
        cfg = self.config
        needs_device_tail = device_out or output_type == "yuv"
        if needs_device_tail and not (self._w_sharded and os.environ.get("DIFFVAE_DEVICE_UNPATCHIFY") == "1"):
            msg = "device_out/yuv need the W-sharded fast path with DIFFVAE_DEVICE_UNPATCHIFY=1"
            raise ValueError(msg)
        if self._w_sharded:
            # ``out`` is (1, batch, T*H*(W/sp), padded_pc): W-sharded over sp_axis, and REPLICATED
            # over the other mesh axis (nothing shards it there -- the input was uploaded replicated
            # on that axis). A composer pull would DMA all `other`x-redundant replicas over PCIe.
            # Instead pull like the Wan/LTX VAEs do: make every device hold a UNIQUE shard, then
            # fast_device_to_host reads a different 1/(sp*other) piece from each device concurrently
            # over all PCIe links. mesh_partition H over the replicated axis is the "keep only my
            # portion" op -- a comms-free local slice (sub-ms), the same trick fast_device_to_host's
            # own multi-host branch uses. Reassembled by mesh coordinate into the (t, h, w) volume.
            sp = int(list(self.mesh_device.shape)[self.sp_axis])
            other_axis = 1 - self.sp_axis
            other = int(list(self.mesh_device.shape)[other_axis])
            w_local = grid.w // sp
            padded_pc = self.padded_patch_channels
            # fast_device_to_host needs a 2D mesh and each device unique along a concatenated axis;
            # only shard the replicated axis when H splits evenly over it, else fall back to the
            # (correct but replica-pulling) composer path.
            can_fast = len(tuple(self.mesh_device.shape)) == 2 and (other == 1 or grid.h % other == 0)
            if can_fast:
                with stage_timer(self.mesh_device, "stage5 tail: device->host pull", category=decode_tree.HOST_XFER):
                    rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.deallocate(out)
                    vol = ttnn.reshape(rm, (1, grid.t, grid.h, w_local, padded_pc))
                    concat_dims = [None, None]
                    concat_dims[self.sp_axis] = 3  # W-band from each sp device
                    shard_other = other > 1
                    if shard_other:
                        vol = ttnn.mesh_partition(vol, dim=2, cluster_axis=other_axis)  # H over the replicated axis
                        concat_dims[other_axis] = 2
                    # DIFFVAE_TRIM_PAD_CHANNELS=1 drops the tile padding BEFORE the pull. conv_out emits
                    # padded_patch_channels because patch_channels (48) is not tile-aligned, so a quarter
                    # of what crosses PCIe is zeros. Trimming on device is exact -- those columns come
                    # from the zero-padded rows of the conv_out weight -- but it is a strided row-major
                    # copy of the whole volume, so whether it wins is a question for the profile.
                    if os.environ.get("DIFFVAE_TRIM_PAD_CHANNELS") == "1":
                        shape = list(vol.shape)
                        trimmed = ttnn.slice(vol, [0] * len(shape), shape[:-1] + [cfg.patch_channels])
                        ttnn.deallocate(vol)
                        vol = trimmed
                    if os.environ.get("DIFFVAE_DEVICE_UNPATCHIFY") == "1":
                        # Depth-to-space on device so the pull lands final-shaped pixels and the host
                        # does nothing. Packed channel order is (c, w_sub, h_sub), per patchify.
                        pv = cfg.patch_size
                        shp = list(vol.shape)
                        wl = shp[3]
                        vol = ttnn.reshape(vol, (1, shp[1], shp[2], wl, cfg.out_channels, pv, pv))
                        vol = ttnn.permute(vol, (0, 4, 1, 2, 6, 3, 5))  # (1, C, T, H, h_sub, W, w_sub)
                        vol = ttnn.reshape(vol, (1, cfg.out_channels, shp[1], shp[2] * pv, wl * pv))
                        concat_dims = [None, None]
                        concat_dims[self.sp_axis] = 4  # W-band, now in pixels
                        if shard_other:
                            concat_dims[other_axis] = 3
                        if device_out:
                            # Everything above is device work a trace can hold; the pull below is not.
                            if shard_other:
                                ttnn.deallocate(rm)
                            return vol
                        if output_type == "yuv":
                            # vol is exactly what the YUV kernel wants: (1, 3, T, H, W) bf16 row-major
                            # in [-1, 1], sharded {mesh axis 0: H, mesh axis 1: W}. Convert and gather
                            # on device so the pull moves 1.5 bytes/pixel instead of 6.
                            h_out, w_out = grid.h * pv, grid.w * pv
                            planar = fast_device_to_host_yuv(
                                vol,
                                self.mesh_device,
                                ccl_manager=self.ccl_manager,
                                logical_h=h_out,
                                logical_w=w_out,
                            )
                            ttnn.deallocate(vol)
                            if shard_other:
                                ttnn.deallocate(rm)
                            return planar.reshape(planar.shape[0], h_out * 3 // 2, w_out)
                        px = fast_device_to_host(vol, self.mesh_device, concat_dims, ccl_manager=self.ccl_manager)
                        ttnn.deallocate(vol)
                        if shard_other:
                            ttnn.deallocate(rm)
                        return px
                    gathered = fast_device_to_host(vol, self.mesh_device, concat_dims, ccl_manager=self.ccl_manager)[
                        ..., : cfg.patch_channels
                    ]  # (1, T, H, W, patch_channels)
                    if shard_other:
                        ttnn.deallocate(rm)
                    ttnn.deallocate(vol)
                with stage_timer(self.mesh_device, "stage5 tail: host unpatchify", category=decode_tree.HOST_COMPUTE):
                    return unpatchify(gathered.permute(0, 4, 1, 2, 3), cfg.patch_size)

            with stage_timer(self.mesh_device, "stage5 tail: device->host pull", category=decode_tree.HOST_XFER):
                gathered = gathered_to_torch(out, mesh_axes=[None, None, self.sp_axis, None])[..., : cfg.patch_channels]
                ttnn.deallocate(out)
            with stage_timer(self.mesh_device, "stage5 tail: host unpatchify", category=decode_tree.HOST_COMPUTE):
                packed = (
                    gathered.reshape(sp, grid.t, grid.h, w_local, cfg.patch_channels)
                    .permute(1, 2, 0, 3, 4)
                    .reshape(grid.batch, grid.t, grid.h, grid.w, cfg.patch_channels)
                    .permute(0, 4, 1, 2, 3)
                )
                return unpatchify(packed, cfg.patch_size)

        # This tensor is replicated across the mesh, so a bare ttnn.to_torch sees one buffer per
        # device and fails. Reading a single chip's copy is what that replication means; the
        # composing helper instead pulls all 32 copies to the host and indexes one out of them,
        # which for a 418 MB output is 13 GB over PCIe and was 100s of a 190s decode.
        with stage_timer(self.mesh_device, "stage5 tail: device->host pull", category=decode_tree.HOST_XFER):
            packed = local_device_to_torch(out)[..., : cfg.patch_channels]
            ttnn.deallocate(out)
        with stage_timer(self.mesh_device, "stage5 tail: host unpatchify", category=decode_tree.HOST_COMPUTE):
            packed = packed.reshape(grid.batch, grid.t, grid.h, grid.w, cfg.patch_channels)
            packed = packed.permute(0, 4, 1, 2, 3)
            return unpatchify(packed, cfg.patch_size)
