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

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
from loguru import logger

import ttnn

from ...layers.embeddings import TimestepEmbedding, Timesteps
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.na3d import neighborhood_attention_3d as na3d_on_device
from ...layers.na3d import neighborhood_attention_3d_op_sp_w_sharded, window_bounds
from ...layers.normalization import RMSNorm
from ...utils.tensor import from_torch as sharded_from_torch
from ...utils.tensor import local_device_to_torch
from ...utils.tensor import to_torch as gathered_to_torch

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

    def frames(self, lo: int, hi: int) -> _RopeTables:
        """The same tables restricted to frames ``[lo, hi)``, for a slab of the volume."""
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
    """
    swapped = ttnn.matmul(x, pair_swap, compute_kernel_config=compute_kernel_config)
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
    """A frame band of the volume: interior ``[lo, hi)`` plus the halo attention reaches into."""

    lo: int
    hi: int
    pad_lo: int
    pad_hi: int

    @property
    def frames(self) -> int:
        return self.hi - self.lo

    @property
    def pad_frames(self) -> int:
        return self.pad_hi - self.pad_lo


def _bands(t: int, *, frames: int | None, kernel: int) -> tuple[_Band, ...]:
    """Split ``t`` frames into bands of ``frames``, each with the halo its windows reach into.

    ``frames=None``, or a band long enough to cover everything, gives one band whose halo is
    empty: every slice downstream is then a no-op and the volume runs whole, which is what short
    videos and the parity tests do.

    The halo comes from :func:`window_bounds`, the rule the attention plan itself is built from,
    rather than from half the kernel: a query within half a kernel of either end has its window
    shifted inward instead of truncated, so it reaches as far as ``kernel - 1`` frames the other
    way. Taking the bound from the shared function also means a band's local windows are the global
    ones shifted by ``pad_lo``, so the attention masks a band builds are the volume's own.
    """
    if frames is None or frames >= t:
        return (_Band(0, t, 0, t),)
    starts, ends = window_bounds(t, kernel)
    bands = []
    for lo in range(0, t, frames):
        hi = min(lo + frames, t)
        bands.append(_Band(lo, hi, starts[lo], ends[hi - 1]))
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
        na3d_backend: str | None = None,
        sp_axis: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        # Backend for the NA3D call: "gather"/"op" run the attention replicated (whole volume on
        # every chip); "op_sp_w_sharded" keeps this chip's W-shard of the sequence through the whole
        # attention (K/V gathered internally), for full-stage spatial-W SP. Defaults to the env
        # override so the whole decoder can be flipped to the op backend for the OOM diagnosis.
        self.na3d_backend = na3d_backend or os.environ.get("DIFFVAE_NA3D_BACKEND", "gather")
        self.sp_axis = sp_axis
        self.scale = config.head_dim**-0.5

        linear = {"bias": True, "mesh_device": mesh_device, "dtype": dtype}
        self.to_q = Linear(config.dim, config.dim, **linear)
        self.to_k = Linear(config.dim, config.dim, **linear)
        self.to_v = Linear(config.dim, config.dim, **linear)
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

    def forward(self, y: ttnn.Tensor, grid: Grid, tables: _RopeTables) -> ttnn.Tensor:
        """``y``: ``(1, batch, sites, dim)``. Returns the same shape.

        ``grid`` is always the FULL ``(T, H, W)``. Under spatial-W SP (``op_sp_w_sharded``) ``y`` is
        this chip's W-shard, so the local W extent is ``W/sp``; the shapes below use that while the
        attention is still told the full W (its executor gathers the missing columns). ``tables``
        must be W-sharded to match ``y`` in that mode (frame piece over this chip's H×(W/sp) rows).
        """
        cfg = self.config
        assert grid.batch == 1, f"batched stage 5 is not implemented; got batch={grid.batch}"
        sharded = self.na3d_backend == "op_sp_w_sharded"
        if sharded:
            sp = int(list(self.mesh_device.shape)[self.sp_axis])
            assert grid.w % sp == 0, f"W={grid.w} must split evenly over sp={sp}"
            w_local = grid.w // sp
        else:
            w_local = grid.w
        sites_local = grid.t * grid.h * w_local
        # Frames are a separate axis rather than folded into the rows, which is what lets the RoPE
        # pieces broadcast: the H/W piece over frames, the T piece over the rows within one.
        heads_shape = (1, grid.t, grid.h * w_local * cfg.num_heads, cfg.head_dim)
        volume_shape = (grid.batch, grid.t, grid.h, w_local, cfg.num_heads, cfg.head_dim)

        def to_volume(x: ttnn.Tensor) -> ttnn.Tensor:
            """Untilize into the volume shape NA3D gathers from, consuming ``x``."""
            out = _reshape_row_major(x, volume_shape)
            if out is not x:
                ttnn.deallocate(x)
            return out

        # Built and consumed one at a time. Holding q, k and v plus each one's untilized copy
        # and RoPE temporaries is what exhausts DRAM at full resolution.
        q = to_volume(
            self._rope(self._normed(self.q_norm, self._projected(self.to_q, y, heads_shape), scale=self.scale), tables)
        )
        k = to_volume(self._rope(self._normed(self.k_norm, self._projected(self.to_k, y, heads_shape)), tables))
        v = to_volume(self._projected(self.to_v, y, heads_shape))

        if sharded:
            out = neighborhood_attention_3d_op_sp_w_sharded(
                q,
                k,
                v,
                dims=(grid.t, grid.h, grid.w),
                kernel_size=cfg.kernel_size,
                sp_axis=self.sp_axis,
                ccl_manager=self.ccl_manager,
                scale=1.0,
            )
        else:
            out = neighborhood_attention_3d(
                q,
                k,
                v,
                kernel_size=cfg.kernel_size,
                scale=1.0,
                ccl_manager=self.ccl_manager,
                backend=self.na3d_backend,
            )
        for tensor in (q, k, v):
            ttnn.deallocate(tensor)

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
        na3d_backend: str | None = None,
        sp_axis: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        # Resolve the backend once so the block and its attention agree on whether the sequence is
        # W-sharded: under "op_sp_w_sharded" the per-chip tensor holds only H*(W/sp) rows per frame,
        # so the block's own frame slicing must use the local rows-per-frame, not the full W.
        self.na3d_backend = na3d_backend or os.environ.get("DIFFVAE_NA3D_BACKEND", "gather")
        self.sp_axis = sp_axis
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
            na3d_backend=self.na3d_backend,
            sp_axis=sp_axis,
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
        if self.na3d_backend == "op_sp_w_sharded":
            sp = int(list(self.mesh_device.shape)[self.sp_axis])
            rows = grid.h * (grid.w // sp)
        else:
            rows = grid.h * grid.w
        # A local view of the volume so the caller's list is left alone; entries become None as
        # this loop releases them.
        live: list[ttnn.Tensor | None] = list(x)
        out: list[ttnn.Tensor] = []
        for index, band in enumerate(bands):
            # Bands own nothing they were handed: ``live`` is freed by this loop's own bookkeeping
            # below, so anything derived from it is released here the moment it stops being read.
            padded = self._padded_rows(live, index, bands, rows)
            interior = ((band.lo - band.pad_lo) * rows, (band.hi - band.pad_lo) * rows)

            context_rows = _slice_rows(context, band.pad_lo * rows, band.pad_hi * rows)
            injected = self.context_proj(context_rows)
            if context_rows is not context:
                ttnn.deallocate(context_rows)
            xs = ttnn.add(padded, injected)
            ttnn.deallocate(injected)
            if padded is not live[index]:
                ttnn.deallocate(padded)

            modulated = _modulate_consuming(self.norm1(xs), scale_msa, shift_msa)
            attended = self.attn(
                modulated,
                Grid(grid.batch, band.pad_frames, grid.h, grid.w),
                tables[index],
            )
            ttnn.deallocate(modulated)

            residual = _slice_rows(xs, *interior)
            if residual is not xs:
                ttnn.deallocate(xs)
            cropped = _slice_rows(attended, *interior)
            if cropped is not attended:
                ttnn.deallocate(attended)
            y = _add_consuming(residual, cropped)

            modulated = _modulate_consuming(self.norm2(y), scale_mlp, shift_mlp)
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
    ) -> ttnn.Tensor:
        """Band ``index``'s rows plus its halo, read out of whichever bands the halo spans."""
        band = bands[index]
        parts = []
        for other, source in enumerate(bands):
            lo = max(band.pad_lo, source.lo)
            hi = min(band.pad_hi, source.hi)
            if lo < hi:
                assert live[other] is not None, f"band {other} was released before band {index} read it"
                parts.append(_slice_rows(live[other], (lo - source.lo) * rows, (hi - source.lo) * rows))
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
        na3d_backend: str | None = None,
        sp_axis: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DiffVAEStage5Config()
        cfg = self.config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.dtype = dtype
        self.modulation_dtype = modulation_dtype
        # Spatial-W SP: when the backend is "op_sp_w_sharded" the whole stage keeps its sequence
        # W-sharded -- context and x_t are uploaded/resharded over W, the blocks run 1/sp, and the
        # tail output is gathered back over W in forward. Defaults to the env override (so the OOM
        # diagnostic can flip the whole decoder to the replicated op backend).
        self.na3d_backend = na3d_backend or os.environ.get("DIFFVAE_NA3D_BACKEND", "gather")
        self.sp_axis = sp_axis
        self._w_sharded = self.na3d_backend == "op_sp_w_sharded"
        if self._w_sharded:
            assert sp_axis is not None, "op_sp_w_sharded needs sp_axis"
            assert ccl_manager is not None, "op_sp_w_sharded needs a ccl_manager"
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
                na3d_backend=self.na3d_backend,
                sp_axis=sp_axis,
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

    def rope_tables(self, grid: Grid) -> _RopeTables:
        tables = self._rope_cache.get(grid)
        if tables is None:
            w_shard = (int(list(self.mesh_device.shape)[self.sp_axis]), self.sp_axis) if self._w_sharded else None
            tables = _build_rope_tables(
                grid,
                dim_split=self.config.resolved_rope_dim_split,
                base=self.config.rope_base,
                num_heads=self.config.num_heads,
                mesh_device=self.mesh_device,
                dtype=self.dtype,
                w_shard=w_shard,
            )
            self._rope_cache[grid] = tables
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
        return _bands(grid.t, frames=int(frames) if frames else None, kernel=kernel)

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
    ) -> ttnn.Tensor:
        """One stage-5 step. Returns padded patch channels at ``(1, batch, sites, ·)``.

        Upstream carries context and x as one ``[context | conv_in_x_t(x)]`` buffer of
        ``2 * dim`` channels and slices it here. They are kept apart instead: both halves are
        tile-aligned, so building the joint buffer and splitting it straight back is an exact
        round trip, and at 1920x1088 it costs 3.3 GB plus two full-size copies for nothing. No
        block writes the context half, so nothing depends on them being adjacent.
        """
        cfg = self.config
        scaled_t = ttnn.multiply(timestep, cfg.timestep_scale_multiplier)
        modulation = self.shared_adaln(self.t_embedder(scaled_t), grid.batch)
        tables = self.rope_tables(grid)
        band_tables = tuple(tables.frames(band.pad_lo, band.pad_hi) for band in bands)
        log_dram(self.mesh_device, f"stage5 entry ({len(bands)} band(s))")
        for index, block in enumerate(self.diff_blocks):
            x = block(x, context, modulation, grid, bands, band_tables)
            log_dram(self.mesh_device, f"stage5 block {index}")

        # The tail runs per band too: its output is a quarter the width of the volume it comes
        # from, so joining after the projection rather than before is the cheap order.
        tail = []
        for band in x:
            tail.append(self.conv_out(self.norm_out(band)))
            ttnn.deallocate(band)
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
        """
        dim = int(context.shape[-1])
        vol = ttnn.reshape(ttnn.to_layout(context, ttnn.ROW_MAJOR_LAYOUT), (1, grid.t, grid.h, grid.w, dim))
        band = ttnn.mesh_partition(vol, dim=3, cluster_axis=self.sp_axis)  # (1, T, H, W/sp, dim)
        w_local = grid.w // int(list(self.mesh_device.shape)[self.sp_axis])
        flat = ttnn.reshape(band, (1, grid.batch, grid.t * grid.h * w_local, dim))
        return ttnn.to_layout(flat, ttnn.TILE_LAYOUT)

    def forward(
        self,
        context: ttnn.Tensor,
        x_t: torch.Tensor,
        timestep: ttnn.Tensor,
        grid: Grid,
    ) -> torch.Tensor:
        """Return pixels. Valid as the whole decode only for ``model_output_type="x0"``
        with a single inference step, which is what the shipped 2.5 DiffVAE config asks for.
        """
        cfg = self.config
        bands = self.bands(grid)
        if self._w_sharded:
            context = self._wshard_context(context, grid)
        out = self.forward_diff_step(context, self.embed_x_t(x_t, bands), timestep, grid, bands)

        if self._w_sharded:
            # ``out`` is this chip's W-band; gather every band to the host and undo the
            # (device, t, h, w_local) reordering back to the (t, h, w) volume before unpatchify.
            sp = int(list(self.mesh_device.shape)[self.sp_axis])
            w_local = grid.w // sp
            gathered = gathered_to_torch(out, mesh_axes=[None, None, self.sp_axis, None])[..., : cfg.patch_channels]
            ttnn.deallocate(out)
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
        packed = local_device_to_torch(out)[..., : cfg.patch_channels]
        ttnn.deallocate(out)
        packed = packed.reshape(grid.batch, grid.t, grid.h, grid.w, cfg.patch_channels)
        packed = packed.permute(0, 4, 1, 2, 3)
        return unpatchify(packed, cfg.patch_size)
