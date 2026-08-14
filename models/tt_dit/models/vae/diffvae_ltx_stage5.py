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
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch

import ttnn

from ...layers.embeddings import TimestepEmbedding, Timesteps
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import RMSNorm

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


def _na3d_host_shim(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    scale: float,
) -> ttnn.Tensor:
    """Run neighborhood attention on the host until a device primitive exists.

    Delegates to upstream's own eager NA3D so a parity failure can only come from the
    ttnn arithmetic around it, never from a second reimplementation of the window
    geometry. That makes ``ltx_core`` an import-time requirement of this path only --
    once ``layers.na3d`` lands, nothing here is reached.
    """
    from ltx_core.model.video_vae.transformer.fallback_na.eager import na3d  # noqa: PLC0415

    device = q.device()
    dtype = q.dtype
    out = na3d(
        ttnn.to_torch(q),
        ttnn.to_torch(k),
        ttnn.to_torch(v),
        kernel_size=kernel_size,
        scale=scale,
    )
    batch, t, h, w, num_heads, head_dim = out.shape
    out = out.reshape(batch, t, h, w, num_heads * head_dim)
    return ttnn.from_torch(out, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)


def neighborhood_attention_3d(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    scale: float = 1.0,
) -> ttnn.Tensor:
    """3D neighborhood attention over ``(B, T, H, W, num_heads, head_dim)`` tensors.

    Q/K/V arrive already RMS-normed, RoPE'd and (for Q) pre-scaled, so ``scale`` is 1.0.
    Returns ``(B, T, H, W, num_heads * head_dim)``. Mirrors upstream's
    ``NAAttentionCallable`` contract.

    Dispatches to ``layers.na3d`` when that module exists and falls back to a host
    round trip otherwise; this is the single swap point for the device primitive.
    """
    try:
        from ...layers.na3d import neighborhood_attention_3d as impl  # noqa: PLC0415
    except ImportError:
        return _na3d_host_shim(q, k, v, kernel_size=kernel_size, scale=scale)
    return impl(q, k, v, kernel_size=kernel_size, scale=scale)


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


class _RopeTables(NamedTuple):
    cos: ttnn.Tensor
    sin: ttnn.Tensor


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
) -> _RopeTables:
    """Fused (T, H, W) absolute-RoPE cos/sin at ``(1, 1, sites * num_heads, head_dim)``.

    Upstream rotates in W-slabs of ``rope_num_tiles`` with running absolute offsets, which
    is arithmetically one full-volume rotation over positions ``0..W-1``; the slabbing is
    a Dynamo-shape concern only.
    """
    d_t, d_h, d_w = dim_split
    head_dim = d_t + d_h + d_w
    positions = (
        torch.arange(grid.t, dtype=torch.float32),
        torch.arange(grid.h, dtype=torch.float32),
        torch.arange(grid.w, dtype=torch.float32),
    )
    inv = (
        _rope_inv_freqs(d_t, base),
        _rope_inv_freqs(d_h, base),
        _rope_inv_freqs(d_w, base),
    )
    # angles[t, h, w, j] for pair j; axis chunks are contiguous in head_dim, so the
    # global pair index j maps to lanes (2j, 2j+1) with no reordering.
    per_axis = [
        pos.reshape(*([1] * axis), -1, *([1] * (2 - axis)), 1) * freq.reshape(1, 1, 1, -1)
        for axis, (pos, freq) in enumerate(zip(positions, inv, strict=True))
    ]
    angles = torch.cat(
        [chunk.expand(grid.t, grid.h, grid.w, chunk.shape[-1]) for chunk in per_axis],
        dim=-1,
    )

    def table(fn) -> ttnn.Tensor:
        pairs = fn(angles).repeat_interleave(2, dim=-1).reshape(grid.sites, 1, head_dim)
        flat = pairs.repeat(1, num_heads, 1).reshape(1, 1, grid.sites * num_heads, head_dim)
        return ttnn.from_torch(flat.contiguous(), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype)

    return _RopeTables(cos=table(torch.cos), sin=table(torch.sin))


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
    rm = ttnn.reshape(rm, shape)
    return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)


def _slice_last(x: ttnn.Tensor, start: int, stop: int) -> ttnn.Tensor:
    """Slice the channel dim. Callers keep ``start``/``stop`` tile-aligned."""
    starts = [0] * (len(x.shape) - 1) + [start]
    stops = [*list(x.shape)[:-1], stop]
    return ttnn.slice(x, starts, stops)


def _modulate(x: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor) -> ttnn.Tensor:
    """``x * (1 + scale) + shift``. ``scale``/``shift`` broadcast over the site axis."""
    return ttnn.add(ttnn.multiply(x, ttnn.add(scale, 1.0)), shift)


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
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
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
        swapped = ttnn.matmul(x, self.pair_swap, compute_kernel_config=self.swap_compute_config)
        return ttnn.add(ttnn.multiply(x, tables.cos), ttnn.multiply(swapped, tables.sin))

    def forward(self, y: ttnn.Tensor, grid: Grid, tables: _RopeTables) -> ttnn.Tensor:
        """``y``: ``(1, batch, sites, dim)``. Returns the same shape."""
        cfg = self.config
        heads_shape = (1, grid.batch, grid.sites * cfg.num_heads, cfg.head_dim)
        volume_shape = (grid.batch, grid.t, grid.h, grid.w, cfg.num_heads, cfg.head_dim)

        q = _reshape_retiled(self.to_q(y), heads_shape)
        k = _reshape_retiled(self.to_k(y), heads_shape)
        v = _reshape_retiled(self.to_v(y), heads_shape)

        q = self._rope(ttnn.multiply(self.q_norm(q), self.scale), tables)
        k = self._rope(self.k_norm(k), tables)

        out = neighborhood_attention_3d(
            _reshape_retiled(q, volume_shape),
            _reshape_retiled(k, volume_shape),
            _reshape_retiled(v, volume_shape),
            kernel_size=cfg.kernel_size,
            scale=1.0,
        )
        out = _reshape_retiled(out, (1, grid.batch, grid.sites, cfg.dim))
        return self.proj(out)


class DiffusionNABlock(Module):
    """Context injection, then AdaLN residual attention, then AdaLN residual SwiGLU."""

    def __init__(
        self,
        config: DiffVAEStage5Config,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType,
    ) -> None:
        super().__init__()
        self.config = config
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
        self.attn = _NeighborhoodAttention3D(config, mesh_device=mesh_device, dtype=dtype)
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
        x: ttnn.Tensor,
        context: ttnn.Tensor,
        shared_modulation: ttnn.Tensor,
        grid: Grid,
        tables: _RopeTables,
    ) -> ttnn.Tensor:
        """``x``/``context``: ``(1, batch, sites, dim)``. Returns the updated ``x`` half."""
        dim = self.config.dim
        # Adding the table to all 7 chunks and then reading 4 is upstream's
        # _modulation: the gate chunks are computed and discarded.
        mod = ttnn.add(shared_modulation, self.scale_shift_table.data)
        scale_msa = _slice_last(mod, 0, dim)
        shift_msa = _slice_last(mod, dim, 2 * dim)
        scale_mlp = _slice_last(mod, 3 * dim, 4 * dim)
        shift_mlp = _slice_last(mod, 4 * dim, 5 * dim)

        x = ttnn.add(x, self.context_proj(context))
        x = ttnn.add(x, self.attn(_modulate(self.norm1(x), scale_msa, shift_msa), grid, tables))
        return ttnn.add(x, self.mlp_down(self.mlp_gate_up(_modulate(self.norm2(x), scale_mlp, shift_mlp))))


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
    ) -> None:
        super().__init__()
        self.config = config or DiffVAEStage5Config()
        cfg = self.config
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.modulation_dtype = modulation_dtype
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
            DiffusionNABlock(cfg, mesh_device=mesh_device, dtype=dtype) for _ in range(cfg.num_blocks)
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
            tables = _build_rope_tables(
                grid,
                dim_split=self.config.resolved_rope_dim_split,
                base=self.config.rope_base,
                num_heads=self.config.num_heads,
                mesh_device=self.mesh_device,
                dtype=self.dtype,
            )
            self._rope_cache[grid] = tables
        return tables

    def embed_x_t(self, x_t: torch.Tensor) -> ttnn.Tensor:
        """Patchify pixel-space ``(B, C, T, H, W)`` noise and project it to ``(1, B, S, dim)``."""
        cfg = self.config
        patched = patchify(x_t, cfg.patch_size)
        batch, _, t, h, w = patched.shape
        flat = patched.permute(0, 2, 3, 4, 1).reshape(1, batch, t * h * w, cfg.patch_channels)
        flat = torch.nn.functional.pad(flat, (0, self.padded_patch_channels - cfg.patch_channels))
        return self.conv_in_x_t(
            ttnn.from_torch(flat.contiguous(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
        )

    def forward_diff_step(
        self,
        context_and_x: ttnn.Tensor,
        timestep: ttnn.Tensor,
        grid: Grid,
    ) -> ttnn.Tensor:
        """One stage-5 step on the block-ready buffer.

        ``context_and_x`` is ``[context | conv_in_x_t(x)]`` at ``(1, batch, sites, 2 * dim)``.
        Splitting it once here rather than per block is equivalent because no block writes
        the context half. Returns padded patch channels at ``(1, batch, sites, ·)``.
        """
        cfg = self.config
        context = _slice_last(context_and_x, 0, cfg.context_channels)
        x = _slice_last(context_and_x, cfg.context_channels, cfg.context_channels + cfg.dim)

        scaled_t = ttnn.multiply(timestep, cfg.timestep_scale_multiplier)
        modulation = self.shared_adaln(self.t_embedder(scaled_t), grid.batch)
        tables = self.rope_tables(grid)
        for block in self.diff_blocks:
            x = block(x, context, modulation, grid, tables)

        return self.conv_out(self.norm_out(x))

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
        context_and_x = ttnn.concat([context, self.embed_x_t(x_t)], dim=-1)
        out = self.forward_diff_step(context_and_x, timestep, grid)

        packed = ttnn.to_torch(out)[..., : cfg.patch_channels]
        packed = packed.reshape(grid.batch, grid.t, grid.h, grid.w, cfg.patch_channels)
        packed = packed.permute(0, 4, 1, 2, 3)
        return unpatchify(packed, cfg.patch_size)
