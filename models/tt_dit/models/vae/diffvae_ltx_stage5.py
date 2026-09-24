# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 DiffVAE decoder stage 5.

Stage 5 denoises patchified noised pixels ``x_t`` under the context volume produced by stages
1-4: a stack of ``DiffusionNABlock``s (3D neighborhood attention plus a SwiGLU MLP) modulated
by a shared AdaLN-Zero projection of the timestep embedding. With ``model_output_type="x0"``
and one inference step, a single pass of :meth:`DiffVAEStage5.forward` returns pixels.

Port of ``ltx_core.model.video_vae``: ``DiffusionVideoDecoder.forward_diff_step`` plus the
``transformer/combined/`` pathway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from loguru import logger

import ttnn

from ...layers.embeddings import LTXAdaLayerNormSingle
from ...layers.feedforward import SwiGLU
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.neighborhood_attention import NAKernel, neighborhood_attention_3d, resolve_na_kernel
from ...layers.neighborhood_attention_plan import window_bounds
from ...layers.neighborhood_permute import (
    SITES_PER_BRICK,
    brick_count,
    brick_grid,
    from_bricked_grid,
    sites_per_t_brick,
    to_bricked_grid,
)
from ...layers.normalization import RMSNorm
from ...utils import timing_tree
from ...utils.ltx import ceil_to
from ...utils.memory_log import log_dram
from ...utils.tensor import fast_device_to_host
from ...utils.tensor import from_torch as sharded_from_torch
from ...utils.tensor import local_device_to_torch
from ...utils.tensor import to_torch as gathered_to_torch
from ...utils.yuv_d2h import fast_device_to_host_yuv
from .diffvae_ops import (
    TILE,
    align_down,
    consume,
    consume_all,
    device_major_qkv,
    mesh_axis_size,
    pad_dim,
    release_intermediates,
    retile,
    slice_last,
    slice_rows,
    split_qkv,
    to_row_major,
    wshard,
)
from .diffvae_rope import ROPE_BASE, default_rope_dim_split, interleaved_lanes, pair_swap_matrix

__all__ = [
    "DiffVAEStage5",
    "DiffVAEStage5Config",
    "Grid",
    "patchify",
    "unpatchify",
]

# AdaLN-Zero chunks in the shared projection. Only 4 are consumed (scale/shift for MSA and MLP);
# the 3 gate chunks exist for checkpoint shape compatibility, the static gates having been folded
# into Linear weights at export.
NUM_ADALN_CHUNKS = 7


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
    # Generalized Neighborhood Attention query-group stride, physical (t,h,w). (1,1,1) is the
    # shipped architecture; anything larger shares one window across each group of queries.
    gna_stride: tuple[int, int, int] = (1, 1, 1)
    context_channels: int = 256
    mlp_hidden: int = 1024
    num_blocks: int = 8
    t_emb_dim: int = 384
    patch_size: int = 4
    out_channels: int = 3
    norm_eps: float = 1e-6
    timestep_scale_multiplier: float = 1000.0
    rope_base: float = ROPE_BASE
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
# Host-side patch packing
# ---------------------------------------------------------------------------


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Space-to-depth on ``(B, C, T, H, W)``; inverse of :func:`unpatchify`.

    Channel order is ``(c, w_sub, h_sub)``: the W sub-index is the *outer* of the two spatial
    ones, matching upstream's ``b c (f p) (h q) (w r) -> b (c p r q) f h w``.
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


class _RopeParts(NamedTuple):
    cos: ttnn.Tensor
    sin: ttnn.Tensor


@dataclass(frozen=True)
class _RopeTables:
    """The RoPE table as two broadcastable pieces, or one fused bricked-order table.

    ``frame`` is ``(1, 1, rows_per_frame, head_dim)`` and carries the H and W lanes; ``time`` is
    ``(1, t, 1, head_dim)`` and carries the T lanes. When ``fused`` is set (bricked order) it is
    the whole rotation and the two pieces are unused.
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
            t_br_hi = ceil_to(hi, self.brick[0]) // self.brick[0]
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


def _build_rope_tables(
    grid: Grid,
    *,
    dim_split: tuple[int, int, int],
    base: float,
    num_heads: int,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
) -> _RopeTables:
    """Absolute (T, H, W) RoPE factored into one frame piece and one row-per-frame piece. Replicated
    only: a W-sharded stage 5 keeps bricked and takes :func:`_build_bricked_rope_tables`.

    A row of the full table is ``[T-lanes(t) | H-lanes(h) | W-lanes(w)]``, so the H and W lanes
    repeat in every frame and the T lanes repeat at every site within a frame. Each piece is zero
    outside its own lanes, so ``cos == frame.cos + time.cos`` exactly and :func:`_apply_rope`
    distributes the multiply over the two instead of materialising a volume-sized table.
    """
    head_dim = sum(dim_split)

    def lanes(fn, axis: int, positions: torch.Tensor) -> torch.Tensor:
        return interleaved_lanes(fn, axis, positions, dim_split, base)

    within = torch.arange(grid.h * grid.w)
    rows_h = torch.div(within, grid.w, rounding_mode="floor")
    rows_w = within % grid.w
    rows_per_frame = grid.h * grid.w * num_heads

    def upload(rows: torch.Tensor, shape: tuple[int, ...]) -> ttnn.Tensor:
        return ttnn.from_torch(
            rows.reshape(shape).contiguous(), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype
        )

    def frame_piece(fn) -> ttnn.Tensor:
        # q and k carry heads inside the row axis, so each site's row repeats once per head.
        rows = (lanes(fn, 1, rows_h) + lanes(fn, 2, rows_w)).reshape(grid.h * grid.w, 1, head_dim)
        full = rows.repeat(1, num_heads, 1)
        return upload(full, (1, 1, grid.h * grid.w * num_heads, head_dim))

    def time_piece(fn) -> ttnn.Tensor:
        return upload(lanes(fn, 0, torch.arange(grid.t)), (1, grid.t, 1, head_dim))

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
    """Fused RoPE in bricked site order, one row per (site, head). The factored frame/time form
    does not survive bricking. Built once per stage, sliced per band on ``T_br``."""
    head_dim = sum(dim_split)
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
            return interleaved_lanes(fn, axis, positions, dim_split, base, ghost=ghost)

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

    With factored tables ``x * cos`` is evaluated as ``x * frame.cos + x * time.cos``, the two
    pieces broadcasting over the frame axis and the row axis respectively. A fused (bricked) table
    is the whole rotation, same shape as ``x``.
    """
    swapped = ttnn.matmul(x, pair_swap, compute_kernel_config=compute_kernel_config)
    if tables.fused is not None:
        aligned = ttnn.multiply(x, tables.fused.cos)
        ttnn.deallocate(x)
        rotated = ttnn.multiply(swapped, tables.fused.sin)
        ttnn.deallocate(swapped)
        return consume_all(ttnn.add, aligned, rotated)
    aligned = consume_all(ttnn.add, ttnn.multiply(x, tables.frame.cos), ttnn.multiply(x, tables.time.cos))
    ttnn.deallocate(x)
    rotated = consume_all(ttnn.add, ttnn.multiply(swapped, tables.frame.sin), ttnn.multiply(swapped, tables.time.sin))
    ttnn.deallocate(swapped)
    return consume_all(ttnn.add, aligned, rotated)


# ---------------------------------------------------------------------------
# Frame bands
# ---------------------------------------------------------------------------


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


def _bands(t: int, *, frames: int | None, kernel: int, align: int = 1) -> tuple[_Band, ...]:
    """Split ``t`` frames into bands of ``frames``, each with the halo its windows reach into.

    ``frames=None``, or a band covering everything, gives one band with an empty halo, so every
    slice downstream is a no-op.

    The halo comes from :func:`window_bounds` rather than half the kernel: a query within half a
    kernel of either end has its window shifted inward instead of truncated, so it reaches as far
    as ``kernel - 1`` frames the other way. A band's local windows are then the global ones
    shifted by ``pad_lo``.

    ``align`` rounds lo / pad_lo down and pad_hi / layout_hi up to a multiple of the brick's T
    extent, so those cuts stay contiguous in bricked order.
    """
    t_layout = ceil_to(t, align)
    if frames is None or frames >= t:
        return (_Band(0, t, 0, t_layout, t_layout),)
    if align > 1:
        frames = max(align, align_down(frames, align))
        if frames >= t:
            return (_Band(0, t, 0, t_layout, t_layout),)
    starts, ends = window_bounds(t, kernel)
    bands = []
    for lo in range(0, t, frames):
        hi = min(lo + frames, t)
        pad_lo, pad_hi = starts[lo], ends[hi - 1]
        if align > 1:
            pad_lo = align_down(pad_lo, align)
            pad_hi = min(ceil_to(pad_hi, align), t_layout)
        layout_hi = min(ceil_to(hi, align), t_layout)
        bands.append(_Band(lo, hi, pad_lo, pad_hi, layout_hi))
    return tuple(bands)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


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
        fused_qkv: bool = False,
        tp_proj: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        # Resolved by the stage and handed down, so the three levels cannot pick different backends.
        self.kernel = resolve_na_kernel(na3d_backend)
        self.sp_axis = sp_axis
        # TP-over-heads on a second mesh axis, only meaningful under a w_sharded kernel.
        self.tp_axis = tp_axis
        self.scale = config.head_dim**-0.5

        # tp_proj: column-parallel qkv over the TP axis, so each chip computes only its heads' q/k/v
        # and feeds the attention already head-sharded. The out-proj stays replicated.
        self.tp_proj = tp_proj and tp_axis is not None
        tp = mesh_axis_size(mesh_device, tp_axis) if self.tp_proj else 1
        assert not self.tp_proj or config.num_heads % tp == 0, f"num_heads={config.num_heads} not divisible by tp={tp}"
        self.heads_local = config.num_heads // tp

        linear = {"bias": True, "mesh_device": mesh_device, "dtype": dtype}
        qkv_linear = dict(linear)
        if self.tp_proj:
            qkv_linear["weight_mesh_axes"] = [None, tp_axis]
            qkv_linear["bias_mesh_axes"] = [None, tp_axis]
        # fused_qkv: one fused qkv matmul, split by slicing the packed output.
        self.fused_qkv = fused_qkv
        if self.fused_qkv:
            self.qkv = Linear(config.dim, 3 * config.dim, **qkv_linear)
        else:
            self.to_q = Linear(config.dim, config.dim, **qkv_linear)
            self.to_k = Linear(config.dim, config.dim, **qkv_linear)
            self.to_v = Linear(config.dim, config.dim, **qkv_linear)
        self.proj = Linear(config.dim, config.dim, **linear)

        norm = {
            "norm_eps": config.norm_eps,
            "bias": False,
            "mesh_device": mesh_device,
            "dtype": dtype,
        }
        self.q_norm = RMSNorm(config.head_dim, **norm)
        self.k_norm = RMSNorm(config.head_dim, **norm)

        self.pair_swap = ttnn.from_torch(
            pair_swap_matrix(config.head_dim),
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
        # Checkpoints ship one Linear(dim, 3*dim) under qkv.*, split [q | k | v] along the output dim.
        # Kept fused, it is regrouped device-major so each chip's column shard is its own q, k and v.
        devices = self.config.num_heads // self.heads_local
        for leaf in ("weight", "bias"):
            fused = state.pop(f"qkv.{leaf}", None)
            if fused is None:
                continue
            if self.fused_qkv:
                state[f"qkv.{leaf}"] = device_major_qkv(fused, devices)
            else:
                state[f"to_q.{leaf}"], state[f"to_k.{leaf}"], state[f"to_v.{leaf}"] = split_qkv(fused)

    def _rope(self, x: ttnn.Tensor, tables: _RopeTables) -> ttnn.Tensor:
        """**Consumes** ``x``."""
        return _apply_rope(x, tables, pair_swap=self.pair_swap, compute_kernel_config=self.swap_compute_config)

    def _normed(self, norm, x: ttnn.Tensor, *, scale: float | None = None) -> ttnn.Tensor:
        """``norm(x)``, optionally scaled, consuming ``x``."""
        out = consume(x, norm)
        return out if scale is None else consume(out, ttnn.multiply, scale)

    def forward(
        self, y: ttnn.Tensor, grid: Grid, tables: _RopeTables, brick: tuple[int, int, int] | None = None
    ) -> ttnn.Tensor:
        """``y``: ``(1, batch, sites, dim)``. Returns the same shape.

        ``grid`` is always the FULL ``(T, H, W)``. Under a W-sharded kernel ``y`` is this chip's
        W-shard, so the local W extent is ``W/sp`` while the attention is told the full W.
        ``tables`` must be W-sharded to match.

        ``brick`` set means ``y`` is already in bricked site order (keep-bricked): Q/K/V stay
        bricked, RoPE uses the fused bricked table, and the op is told ``already_bricked``.
        """
        cfg = self.config
        assert grid.batch == 1, f"batched stage 5 is not implemented; got batch={grid.batch}"
        sharded = self.kernel.w_sharded
        if sharded:
            sp = mesh_axis_size(self.mesh_device, self.sp_axis)
            assert grid.w % sp == 0, f"W={grid.w} must split evenly over sp={sp}"
            w_local = grid.w // sp
        else:
            w_local = grid.w
        sites_local = (
            brick_count((grid.t, grid.h, w_local), brick) * SITES_PER_BRICK
            if brick is not None
            else grid.t * grid.h * w_local
        )
        # Frames are a separate axis so the factored RoPE pieces can broadcast. Bricked RoPE is
        # fused (one row per site and head), so the T axis collapses.
        heads = self.heads_local
        heads_shape = (
            (1, 1, sites_local * heads, cfg.head_dim)
            if brick is not None
            else (1, grid.t, grid.h * w_local * heads, cfg.head_dim)
        )
        volume_shape = (grid.batch, grid.t, grid.h, w_local, heads, cfg.head_dim)

        def to_volume(x: ttnn.Tensor) -> ttnn.Tensor:
            """Untilize into the volume shape NA3D gathers from, consuming ``x``."""
            return consume(x, to_row_major, volume_shape)

        if brick is not None:
            # A relabel, not a copy: ttnn.reshape hands back a new wrapper over the SAME buffer, so
            # the input must not be deallocated by object identity (see release_intermediates).
            def to_bricked_seq(x: ttnn.Tensor) -> ttnn.Tensor:
                target = (grid.batch, heads, sites_local, cfg.head_dim)
                if tuple(x.shape) == target:
                    return x
                out = ttnn.reshape(x, target)
                release_intermediates((x,), keep=out)
                return out

            prep = to_bricked_seq
        else:
            prep = to_volume

        # Lanes are built and consumed one at a time to bound peak DRAM.
        if self.fused_qkv:
            with timing_tree.span(self.mesh_device, "qkv-proj", category=timing_tree.PROJ, deep=True):
                packed = self.qkv(y)
            width = self.heads_local * cfg.head_dim

            def lane(index: int) -> ttnn.Tensor:
                return consume(slice_last(packed, index * width, (index + 1) * width), retile, heads_shape)

            with timing_tree.span(
                self.mesh_device, "qkv-lanes: slice+norm+rope", category=timing_tree.NORM_ROPE, deep=True
            ):
                q = prep(self._rope(self._normed(self.q_norm, lane(0), scale=self.scale), tables))
                k = prep(self._rope(self._normed(self.k_norm, lane(1)), tables))
                v = prep(lane(2))
                ttnn.deallocate(packed)
        else:

            def lane_unfused(projection, norm, *, scale=None, rope=True):
                with timing_tree.span(self.mesh_device, "qkv-proj", category=timing_tree.PROJ, deep=True):
                    part = consume(projection(y), retile, heads_shape)
                if norm is not None:
                    with timing_tree.span(self.mesh_device, "qkv-norm", category=timing_tree.NORM_ROPE, deep=True):
                        part = self._normed(norm, part, scale=scale)
                    if rope:
                        with timing_tree.span(self.mesh_device, "qkv-rope", category=timing_tree.NORM_ROPE, deep=True):
                            part = self._rope(part, tables)
                with timing_tree.span(
                    self.mesh_device, "qkv-prep (to seq/volume)", category=timing_tree.RESHAPE, deep=True
                ):
                    return prep(part)

            q = lane_unfused(self.to_q, self.q_norm, scale=self.scale)
            k = lane_unfused(self.to_k, self.k_norm)
            v = lane_unfused(self.to_v, None)

        # Under the W-sharded kernel this is this chip's W-shard plus the halo its windows reach
        # into; window placement stays global, so a query near a shard seam still sees a full window.
        out = neighborhood_attention_3d(
            q,
            k,
            v,
            kernel=self.kernel,
            kernel_size=cfg.kernel_size,
            scale=1.0,
            dims=(grid.t, grid.h, grid.w),
            ccl_manager=self.ccl_manager,
            sp_axis=self.sp_axis,
            tp_axis=self.tp_axis,
            heads_presharded=self.tp_proj,
            brick=brick,
            stride=cfg.gna_stride,
        )
        for tensor in (q, k, v):
            ttnn.deallocate(tensor)

        with timing_tree.span(self.mesh_device, "out-proj", category=timing_tree.PROJ, deep=True):
            flat = consume(out, retile, (1, grid.batch, sites_local, cfg.dim))
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
        fused_qkv: bool = False,
        tp_proj: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
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
            fused_qkv=fused_qkv,
            tp_proj=tp_proj,
        )
        self.norm2 = RMSNorm(config.dim, **norm)
        # One fused [up | gate] GEMM whose epilogue emits silu(gate) * up; replicated (TP is over heads).
        self.mlp = SwiGLU(config.dim, config.mlp_hidden, mesh_device=mesh_device, dtype=dtype, fused=True)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "scale_shift_table" in state:
            state["scale_shift_table"] = state["scale_shift_table"].reshape(1, -1)

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
        """``x``: the volume as one ``(1, batch, band rows, dim)`` tensor per band; returns the
        updated volume in the same form. **Consumes** ``x``; ``context`` is left alone.

        Attention is the only part that reads across sites, so each band runs on its own rows and
        only the attention sees the band plus its halo, cropped back afterwards. Because the halo
        is exactly what :func:`window_bounds` reaches, every query kept sees the window it would
        have seen whole and the crop is exact.
        """
        dim = self.config.dim
        # Adding the table to all 7 chunks and then reading 4 is upstream's _modulation.
        mod = ttnn.add(shared_modulation, self.scale_shift_table.data)
        scale_msa = slice_last(mod, 0, dim)
        shift_msa = slice_last(mod, dim, 2 * dim)
        scale_mlp = slice_last(mod, 3 * dim, 4 * dim)
        shift_mlp = slice_last(mod, 4 * dim, 5 * dim)

        # Rows per frame on THIS chip (H*(W/sp) under W-SP). In bricked order the slice unit is one
        # T-brick of sites instead.
        if self.kernel.w_sharded:
            sp = mesh_axis_size(self.mesh_device, self.sp_axis)
            w_local = grid.w // sp
            rows = sites_per_t_brick((grid.t, grid.h, w_local), brick) if brick is not None else grid.h * w_local
        else:
            rows = sites_per_t_brick((grid.t, grid.h, grid.w), brick) if brick is not None else grid.h * grid.w
        frame_step = brick[0] if brick is not None else 1
        # A local view of the volume; entries become None as this loop releases them.
        live: list[ttnn.Tensor | None] = list(x)
        out: list[ttnn.Tensor] = []
        for index, band in enumerate(bands):
            with timing_tree.span(
                self.mesh_device, "halo assemble (padded rows)", category=timing_tree.RESHAPE, deep=True
            ):
                padded = self._padded_rows(live, index, bands, rows, frame_step=frame_step)
            interior = (
                (band.lo - band.pad_lo) // frame_step * rows,
                (band.layout_hi - band.pad_lo) // frame_step * rows,
            )

            xs = self._inject_context(padded, context, band, rows, frame_step, owned=padded is live[index])
            band_grid = Grid(grid.batch, min(band.pad_hi, grid.t) - band.pad_lo, grid.h, grid.w)
            attended = self._attend(
                self._modulated("pre-attn", self.norm1, xs, scale_msa, shift_msa), band_grid, tables[index], brick
            )

            with timing_tree.span(
                self.mesh_device, "residual crop+add (attn)", category=timing_tree.RESHAPE, deep=True
            ):
                residual = consume(xs, slice_rows, *interior)
                cropped = consume(attended, slice_rows, *interior)
                y = consume_all(ttnn.add, residual, cropped)

            out.append(self._mlp(self._modulated("pre-mlp", self.norm2, y, scale_mlp, shift_mlp), y))

            # A band's input rows are read as halo by its neighbours; release them once no band
            # still to come reaches back that far.
            reach = bands[index + 1].pad_lo if index + 1 < len(bands) else bands[-1].pad_hi
            for other, entry in enumerate(live):
                if entry is not None and bands[other].hi <= reach:
                    ttnn.deallocate(entry)
                    live[other] = None
        return out

    @timing_tree.span("mesh_device", "context-inject", category=timing_tree.CONTEXT_INJECT)
    def _inject_context(
        self, padded: ttnn.Tensor, context: ttnn.Tensor, band: _Band, rows: int, frame_step: int, *, owned: bool
    ) -> ttnn.Tensor:
        """``padded`` plus the projected context rows of its frames. **Consumes** ``padded`` unless it is
        a band's own tensor (``owned``), which the caller's bookkeeping still has to read."""
        context_rows = slice_rows(context, band.pad_lo // frame_step * rows, band.pad_hi // frame_step * rows)
        injected = self.context_proj(context_rows)
        if context_rows is not context:
            ttnn.deallocate(context_rows)
        xs = ttnn.add(padded, injected)
        ttnn.deallocate(injected)
        if not owned:
            ttnn.deallocate(padded)
        return xs

    @timing_tree.span(
        "mesh_device", lambda self, phase, *a: f"norm+modulate ({phase})", category=timing_tree.NORM_ROPE, deep=True
    )
    def _modulated(self, phase: str, norm, x: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor) -> ttnn.Tensor:
        """``norm(x) * (1 + scale) + shift``; ``scale``/``shift`` broadcast over the site axis."""
        scaled = consume(norm(x), ttnn.multiply, ttnn.add(scale, 1.0))
        return consume(scaled, ttnn.add, shift)

    @timing_tree.span("mesh_device", "attention", category=timing_tree.ATTENTION)
    def _attend(self, modulated: ttnn.Tensor, band_grid: Grid, tables: _RopeTables, brick) -> ttnn.Tensor:
        """Attention over one padded band. **Consumes** ``modulated``."""
        attended = self.attn(modulated, band_grid, tables, brick=brick)
        ttnn.deallocate(modulated)
        return attended

    @timing_tree.span("mesh_device", "mlp", category=timing_tree.MLP)
    def _mlp(self, modulated: ttnn.Tensor, y: ttnn.Tensor) -> ttnn.Tensor:
        """SwiGLU plus the residual add. **Consumes** ``modulated`` and ``y``."""
        return consume_all(ttnn.add, y, self.mlp(modulated))

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
                    slice_rows(
                        live[other],
                        (lo - source.lo) // frame_step * rows,
                        (hi - source.lo) // frame_step * rows,
                    )
                )
        if len(parts) == 1:
            return parts[0]
        joined = ttnn.concat(parts, dim=-2)
        # A part that is a whole band is that band's own tensor, borrowed; the rest are copies.
        for part in parts:
            if not any(part is entry for entry in live):
                ttnn.deallocate(part)
        return joined


class DiffVAEStage5(Module):
    """The stage-5 diffusion stack: ``forward_diff_step`` plus patch packing."""

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
        fused_qkv: bool = False,
        tp_proj: bool = True,
        slab_frames: int | None = None,
        device_unpatchify: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or DiffVAEStage5Config()
        cfg = self.config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.dtype = dtype
        self.modulation_dtype = modulation_dtype
        # The ONE place the backend is resolved; the blocks and their attention are handed the record.
        self.kernel = resolve_na_kernel(na3d_backend or "linear_order")
        self.sp_axis = sp_axis
        # TP-over-heads on a second mesh axis: only the per-head attention shards over it.
        self.tp_axis = tp_axis
        self._w_sharded = self.kernel.w_sharded
        self._keep_bricked = self.kernel.keep_bricked
        self._brick: tuple[int, int, int] | None = None
        # Frames per band, or None for the whole volume; see bands().
        self.slab_frames = slab_frames
        # Tile-pad trim and depth-to-space on device before the pull; needed for device_out / yuv.
        self.device_unpatchify = device_unpatchify
        # Under column-parallel qkv the RoPE tables (one row per head) are built for the local head count.
        self.tp_proj = tp_proj and tp_axis is not None
        tp = mesh_axis_size(mesh_device, tp_axis) if self.tp_proj else 1
        self._rope_num_heads = self.config.num_heads // tp
        if self._w_sharded:
            assert sp_axis is not None, f"{self.kernel.name} needs sp_axis"
            assert ccl_manager is not None, f"{self.kernel.name} needs a ccl_manager"
        #: Chips along the W-shard axis; 1 when replicated.
        self.sp = mesh_axis_size(mesh_device, sp_axis) if self._w_sharded else 1
        # The tile-aligned width the patch channels are zero-padded to. The pad has to be explicit
        # on conv_in_x_t's K axis: a garbage-filled activation tail would otherwise multiply against
        # whatever the weight's own tile pad holds.
        self.padded_patch_channels = ceil_to(cfg.patch_channels, TILE)
        self._rope_cache: dict[Grid, _RopeTables] = {}

        self.conv_in_x_t = Linear(self.padded_patch_channels, cfg.dim, bias=True, mesh_device=mesh_device, dtype=dtype)
        # Upstream's t_embedder + shared_adaln: sinusoidal timestep -> MLP -> silu -> Linear to the
        # 7 AdaLN chunks. It feeds multiplicative modulation in every block, so it runs at higher precision.
        self.adaln = LTXAdaLayerNormSingle(
            cfg.t_emb_dim, mesh_device=mesh_device, dtype=modulation_dtype, out_features=NUM_ADALN_CHUNKS * cfg.dim
        )
        self.diff_blocks = ModuleList(
            DiffusionNABlock(
                cfg,
                mesh_device=mesh_device,
                dtype=dtype,
                ccl_manager=ccl_manager,
                na3d_backend=self.kernel,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                fused_qkv=fused_qkv,
                tp_proj=tp_proj,
            )
            for _ in range(cfg.num_blocks)
        )
        self.norm_out = RMSNorm(cfg.dim, norm_eps=cfg.norm_eps, bias=False, mesh_device=mesh_device, dtype=dtype)
        self.conv_out = Linear(cfg.dim, self.padded_patch_channels, bias=True, mesh_device=mesh_device, dtype=dtype)

    GATE_SUFFIXES = (".gate_msa", ".gate_mlp", ".gate_ctx")

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Static gates were folded into attn.proj / mlp.w_down / context_proj at export; an
        # unfolded checkpoint would silently decode wrong.
        gates = sorted(k for k in state if k.endswith(self.GATE_SUFFIXES))
        if gates:
            msg = f"unfolded static gates present ({', '.join(gates)}); this port assumes pre-folded weights"
            raise ValueError(msg)

        # Shipped (out, in) weights: conv_in_x_t is padded on its input features, conv_out on its outputs.
        for key, dim in (("conv_in_x_t.weight", 1), ("conv_out.weight", 0), ("conv_out.bias", 0)):
            if key in state:
                state[key] = pad_dim(state[key], dim, self.padded_patch_channels)

        # The checkpoint's t_embedder (a bare Sequential, mlp.0 / mlp.2) and shared_adaln.proj are
        # one LTXAdaLayerNormSingle here.
        renames = {
            "t_embedder.mlp.0.": "adaln.emb.timestep_embedder.linear_1.",
            "t_embedder.mlp.2.": "adaln.emb.timestep_embedder.linear_2.",
            "shared_adaln.proj.": "adaln.linear.",
        }
        for key in list(state):
            for src, dst in renames.items():
                if key.startswith(src):
                    state[dst + key[len(src) :]] = state.pop(key)

    def _stage5_brick(self, grid: Grid) -> tuple[int, int, int]:
        """The brick the whole stage converts with: the same choice the attention op would make."""
        if self._brick is not None:
            return self._brick
        from ...layers.neighborhood_attention_plan import _choose_sharded_brick, brick_override

        volume = (grid.t, grid.h, grid.w)
        context_window = tuple(min(window, extent) for window, extent in zip(self.config.kernel_size, volume))
        self._brick = brick_override(volume) or _choose_sharded_brick(
            volume, context_window, self.config.gna_stride, grid.w // self.sp, self.sp
        )
        return self._brick

    def _local_volume(self, grid: Grid, t: int | None = None) -> tuple[int, int, int]:
        return (grid.t if t is None else t, grid.h, grid.w // self.sp)

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
        # Freed together, by buffer rather than by object: vol is a view of rm and flat one of grid5.
        release_intermediates((x, rm, vol, grid5, flat), keep=out)
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
            w_shard = (self.sp, self.sp_axis) if self._w_sharded else None
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
                assert w_shard is None, "a W-sharded stage 5 keeps bricked; the factored table is replicated only"
                tables = _build_rope_tables(
                    grid,
                    dim_split=self.config.resolved_rope_dim_split,
                    base=self.config.rope_base,
                    num_heads=self._rope_num_heads,
                    mesh_device=self.mesh_device,
                    dtype=self.dtype,
                )
            self._rope_cache[key] = tables
        return tables

    def bands(self, grid: Grid) -> tuple[_Band, ...]:
        """How to split the volume into frame bands of ``slab_frames`` (None runs it whole).

        A band boundary has to be tile-aligned, which holds exactly when ``h * w`` is a multiple of ``TILE``.
        """
        frames = self.slab_frames
        kernel = self.config.kernel_size[0]
        if frames and (grid.h * grid.w) % TILE != 0:
            logger.warning(
                f"[diffvae] ignoring slab_frames={frames}: h*w={grid.h * grid.w} is not a multiple of {TILE}, "
                "so a frame boundary is not a tile boundary"
            )
            frames = None
        align = self._stage5_brick(grid)[0] if self._keep_bricked else 1
        return _bands(grid.t, frames=frames, kernel=kernel, align=align)

    def device_x_t(self, grid: Grid, bands: tuple[_Band, ...], *, seed: int = 0) -> list[ttnn.Tensor]:
        """x_t noise drawn on device, already in the patchified layout. One tensor per band.

        Patchify is a permutation of iid samples, so drawing at the destination layout is the same
        distribution as drawing pixel-space noise and reshuffling it, and the W-shard needs no
        reorder for the same reason. Channels past ``patch_channels`` land on the zero-padded
        columns of ``conv_in_x_t``. Drawn at the FULL volume and partitioned: ttnn.randn replicates
        across the mesh, so a per-chip draw would hand every W-band identical values.
        """
        sp = self.sp
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
        """Patchify pixel-space ``(B, C, T, H, W)`` noise and project it, one tensor per band."""
        cfg = self.config
        patched = patchify(x_t, cfg.patch_size)
        batch = patched.shape[0]
        sp = self.sp
        out = []
        for band in bands:
            rows = patched[:, :, band.lo : band.hi]
            t, h, w = rows.shape[2:]
            flat = rows.permute(0, 2, 3, 4, 1).reshape(1, batch, t * h * w, cfg.patch_channels)
            flat = torch.nn.functional.pad(flat, (0, self.padded_patch_channels - cfg.patch_channels))
            if self._w_sharded:
                # Reorder the (t, h, w) rows to (device, t, h, w_local) so a sharded upload hands
                # device p its W-band.
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

    def modulation(self, timestep: ttnn.Tensor, batch: int) -> ttnn.Tensor:
        """The shared AdaLN-Zero modulation of ``timestep`` ``(1, 1, batch, 1)``, in the modulation dtype.

        Returns ``(1, batch, 1, 7 * dim)`` in the activation dtype: upstream's ``(B, 1, 1, 1, C)``
        view, which the blocks add their own ``scale_shift_table`` to.
        """
        scaled = ttnn.multiply(timestep, self.config.timestep_scale_multiplier)
        mod, _ = self.adaln(scaled)
        if mod.dtype != self.dtype:
            mod = ttnn.typecast(mod, self.dtype)
        return retile(mod, (1, batch, 1, mod.shape[-1]))

    @timing_tree.span("mesh_device", "stage5 diff-blocks (attn+MLP)")
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

        Upstream carries context and x as one ``[context | conv_in_x_t(x)]`` buffer; they are kept
        apart here, which is exact since no block writes the context half.
        """
        with timing_tree.span(self.mesh_device, "stage5 setup: AdaLN + rope tables", category=timing_tree.SETUP):
            modulation = self.modulation(timestep, grid.batch)
            tables = self.rope_tables(grid)
            band_tables = tuple(tables.frames(band.pad_lo, band.pad_hi) for band in bands)
        log_dram(self.mesh_device, f"stage5 entry ({len(bands)} band(s))")
        for index, block in enumerate(self.diff_blocks):
            with timing_tree.span(self.mesh_device, f"stage5 block {index}"):
                x = block(x, context, modulation, grid, bands, band_tables, brick=brick)
            log_dram(self.mesh_device, f"stage5 block {index}")
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

    @timing_tree.span("mesh_device", "stage5 TOTAL (forward)")
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
        """Return pixels. Valid as the whole decode only for ``model_output_type="x0"`` with a
        single inference step, which is what the shipped 2.5 DiffVAE config asks for.

        ``device_out=True`` returns pixels still on device, stopping immediately before the PCIe
        pull, so a caller capturing the decode as a trace can transfer them itself.

        ``context_sharded=True`` means the context arrived W-sharded (this chip's band, same
        ``sp_axis``), so the re-shard is skipped.
        """
        cfg = self.config
        bands = self.bands(grid)
        with timing_tree.span(self.mesh_device, "stage5: context reshard", category=timing_tree.RESHAPE):
            if self._w_sharded and not context_sharded:
                context = wshard(context, (grid.t, grid.h, grid.w), sp_axis=self.sp_axis)
            elif self._w_sharded:
                context = ttnn.to_layout(context, ttnn.TILE_LAYOUT)

        _label = "stage5: device randn + embed x_t" if x_t is None else "stage5: host patchify + embed x_t"
        with timing_tree.span(self.mesh_device, _label, category=timing_tree.HOST_COMPUTE):
            x_bands = self.device_x_t(grid, bands, seed=seed) if x_t is None else self.embed_x_t(x_t, bands)

        brick = self._stage5_brick(grid) if self._keep_bricked else None
        if brick is not None:
            with timing_tree.span(self.mesh_device, "stage5: brick x+context", category=timing_tree.RESHAPE):
                context = self._brick_activation(context, self._local_volume(grid), brick)
                x_bands = [
                    self._brick_activation(band_x, self._local_volume(grid, t=band.hi - band.lo), brick)
                    for band_x, band in zip(x_bands, bands)
                ]

        out = self.forward_diff_step(context, x_bands, timestep, grid, bands, brick=brick)
        return self._to_pixels(out, grid, device_out=device_out, output_type=output_type)

    def pull_pixels(self, vol: ttnn.Tensor, grid: Grid, output_type: str = "float", *, release: bool = True):
        """The PCIe pull of the final pixel volume, host-shaped for the caller.

        ``vol`` is what ``forward(device_out=True)`` returns: ``(1, 3, T, H, W)`` bf16 row-major in
        ``[-1, 1]``, this chip's W-band (and H-band over the other mesh axis when H divides it).
        ``yuv`` converts and gathers YUV 4:2:0 on device first. ``release=False`` keeps ``vol``
        allocated, for a trace whose output buffer it is.
        """
        cfg = self.config
        pv = cfg.patch_size
        other_axis = 1 - self.sp_axis
        other = mesh_axis_size(self.mesh_device, other_axis)
        if output_type == "yuv":
            h_out, w_out = grid.h * pv, grid.w * pv
            planar = fast_device_to_host_yuv(
                vol, self.mesh_device, ccl_manager=self.ccl_manager, logical_h=h_out, logical_w=w_out
            )
            if release:
                ttnn.deallocate(vol)
            return planar.reshape(planar.shape[0], h_out * 3 // 2, w_out)
        concat_dims = [None, None]
        concat_dims[self.sp_axis] = 4
        if other > 1:
            concat_dims[other_axis] = 3
        px = fast_device_to_host(vol, self.mesh_device, concat_dims, ccl_manager=self.ccl_manager)
        if release:
            ttnn.deallocate(vol)
        return px

    def _to_pixels(self, out, grid, *, device_out: bool = False, output_type: str = "float"):
        cfg = self.config
        needs_device_tail = device_out or output_type == "yuv"
        if needs_device_tail and not (self._w_sharded and self.device_unpatchify):
            msg = "device_out/yuv need the W-sharded fast path with device_unpatchify"
            raise ValueError(msg)
        if self._w_sharded:
            # ``out`` is W-sharded over sp_axis and REPLICATED over the other mesh axis. Rather than
            # pull every replica, mesh_partition H over the replicated axis so every device holds a
            # unique shard and fast_device_to_host reads a different piece from each concurrently.
            sp = self.sp
            other_axis = 1 - self.sp_axis
            other = mesh_axis_size(self.mesh_device, other_axis)
            w_local = grid.w // sp
            padded_pc = self.padded_patch_channels
            # fast_device_to_host needs a 2D mesh and H divisible over the replicated axis.
            can_fast = len(tuple(self.mesh_device.shape)) == 2 and (other == 1 or grid.h % other == 0)
            if can_fast:
                with timing_tree.span(
                    self.mesh_device, "stage5 tail: device->host pull", category=timing_tree.HOST_XFER
                ):
                    rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.deallocate(out)
                    vol = ttnn.reshape(rm, (1, grid.t, grid.h, w_local, padded_pc))
                    concat_dims = [None, None]
                    concat_dims[self.sp_axis] = 3
                    shard_other = other > 1
                    if shard_other:
                        vol = ttnn.mesh_partition(vol, dim=2, cluster_axis=other_axis)
                        concat_dims[other_axis] = 2
                    # Drop the tile padding before the pull. Exact, since those columns come from
                    # the zero-padded rows of the conv_out weight.
                    if self.device_unpatchify:
                        shape = list(vol.shape)
                        trimmed = ttnn.slice(vol, [0] * len(shape), shape[:-1] + [cfg.patch_channels])
                        ttnn.deallocate(vol)
                        vol = trimmed
                    if self.device_unpatchify:
                        # Depth-to-space on device. Packed channel order is (c, w_sub, h_sub), per patchify.
                        pv = cfg.patch_size
                        shp = list(vol.shape)
                        wl = shp[3]
                        vol = ttnn.reshape(vol, (1, shp[1], shp[2], wl, cfg.out_channels, pv, pv))
                        vol = ttnn.permute(vol, (0, 4, 1, 2, 6, 3, 5))
                        vol = ttnn.reshape(vol, (1, cfg.out_channels, shp[1], shp[2] * pv, wl * pv))
                        if shard_other:
                            ttnn.deallocate(rm)
                        if device_out:
                            return vol
                        return self.pull_pixels(vol, grid, output_type)
                    gathered = fast_device_to_host(vol, self.mesh_device, concat_dims, ccl_manager=self.ccl_manager)[
                        ..., : cfg.patch_channels
                    ]
                    if shard_other:
                        ttnn.deallocate(rm)
                    ttnn.deallocate(vol)
                with timing_tree.span(
                    self.mesh_device, "stage5 tail: host unpatchify", category=timing_tree.HOST_COMPUTE
                ):
                    return unpatchify(gathered.permute(0, 4, 1, 2, 3), cfg.patch_size)

            with timing_tree.span(self.mesh_device, "stage5 tail: device->host pull", category=timing_tree.HOST_XFER):
                gathered = gathered_to_torch(out, mesh_axes=[None, None, self.sp_axis, None])[..., : cfg.patch_channels]
                ttnn.deallocate(out)
            with timing_tree.span(self.mesh_device, "stage5 tail: host unpatchify", category=timing_tree.HOST_COMPUTE):
                packed = (
                    gathered.reshape(sp, grid.t, grid.h, w_local, cfg.patch_channels)
                    .permute(1, 2, 0, 3, 4)
                    .reshape(grid.batch, grid.t, grid.h, grid.w, cfg.patch_channels)
                    .permute(0, 4, 1, 2, 3)
                )
                return unpatchify(packed, cfg.patch_size)

        # Replicated across the mesh: read one chip's copy rather than composing every replica.
        with timing_tree.span(self.mesh_device, "stage5 tail: device->host pull", category=timing_tree.HOST_XFER):
            packed = local_device_to_torch(out)[..., : cfg.patch_channels]
            ttnn.deallocate(out)
        with timing_tree.span(self.mesh_device, "stage5 tail: host unpatchify", category=timing_tree.HOST_COMPUTE):
            packed = packed.reshape(grid.batch, grid.t, grid.h, grid.w, cfg.patch_channels)
            packed = packed.permute(0, 4, 1, 2, 3)
            return unpatchify(packed, cfg.patch_size)
