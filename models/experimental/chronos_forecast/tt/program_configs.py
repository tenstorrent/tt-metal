# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Explicit matmul program configs for Chronos per-token linears.

The DRAM-interleaved auto-config does not fold a batched activation's batch
into M, so ``(B, T, d) @ (d, n)`` runs as B tiny matmuls on part of the grid.
These 2D multicast configs set ``fuse_batch=True`` (M = B * padded T) and tile
the whole compute grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TILE = 32
# in0 + in1 double-buffered, plus the output block, in bf16 tiles (2 KiB each).
_L1_TILE_BUDGET = 560
# Smaller when activations live in L1, so circular buffers leave room for them.
_L1_RESIDENT_TILE_BUDGET = 320
_MAX_IN0_BLOCK_W = 12
_MAX_OUT_BLOCK_H = 16
_LOFI_L1_MAX_OUT_BLOCK_H = 8
# 1D in1-mcast in0 block caps (sweeps/sweep_matmul_l1.py): HiFi2 peaks at 4, LoFi still gains at 8.
_1D_MAX_IN0_BLOCK_W = 4
_LOFI_1D_MAX_IN0_BLOCK_W = 8


@dataclass(frozen=True)
class TtChronosPrecision:
    """Opt-in reduced precision for the encoder. The default is bf16 with HiFi2 matmuls."""

    bf8_weights: bool = False  # encoder attention + FF weights stored as bfloat8_b
    bf8_attention: bool = False  # Q/K/V (RoPE and SDPA inputs) as bfloat8_b
    bf8_ff_hidden: bool = False  # relu(x @ Wi) stored as bfloat8_b
    bf8_sublayer_out: bool = False  # attention / FF outputs added to the (bf16) residual stream
    bf8_norm_out: bool = False  # RMSNorm outputs (QKV, V*O and FF-up matmul inputs) as bfloat8_b
    lofi_ff: bool = False  # FF-up/down matmuls at LoFi
    lofi_attention: bool = False  # QKV, output and group V*O matmuls at LoFi

    @classmethod
    def performance(cls) -> "TtChronosPrecision":
        return cls(
            bf8_weights=True,
            bf8_attention=True,
            bf8_ff_hidden=True,
            bf8_sublayer_out=True,
            bf8_norm_out=True,
            lofi_ff=True,
            lofi_attention=True,
        )

    def weight_dtype(self):
        import ttnn

        return ttnn.bfloat8_b if self.bf8_weights else ttnn.bfloat16

    def attention_dtype(self):
        import ttnn

        return ttnn.bfloat8_b if self.bf8_attention else None

    def ff_hidden_dtype(self):
        import ttnn

        return ttnn.bfloat8_b if self.bf8_ff_hidden else None

    def sublayer_out_dtype(self):
        import ttnn

        return ttnn.bfloat8_b if self.bf8_sublayer_out else ttnn.bfloat16

    def norm_dtype(self):
        import ttnn

        return ttnn.bfloat8_b if self.bf8_norm_out else None

    def ff_math_fidelity(self):
        import ttnn

        return ttnn.MathFidelity.LoFi if self.lofi_ff else None

    def attention_math_fidelity(self):
        import ttnn

        return ttnn.MathFidelity.LoFi if self.lofi_attention else None

    def l1_chunk_tokens(self) -> int:
        """Largest L1-resident encoder chunk (series x tile-padded tokens) measured to fit a P150."""
        return _L1_CHUNK_TOKENS_BF8 if self.bf8_attention and self.bf8_ff_hidden else _L1_CHUNK_TOKENS_BF16


# 48 / 64 series of 160 padded tokens; the next step up clashes with matmul circular buffers.
_L1_CHUNK_TOKENS_BF16 = 48 * 160
_L1_CHUNK_TOKENS_BF8 = 64 * 160
# (x, y) worker grid of the P150 the chunk budgets were measured on.
L1_CHUNK_GRID = (11, 10)


def compute_kernel_config(math_fidelity=None, *, fp32_dest_acc_en: bool = False, packer_l1_acc: bool = True):
    import ttnn

    from models.common.utility_functions import is_blackhole

    cls = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    return cls(
        math_fidelity=ttnn.MathFidelity.HiFi2 if math_fidelity is None else math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )


def _largest_divisor_at_most(n: int, cap: int) -> int:
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _subblock(block_h: int, block_w: int, max_tiles: int, *, widest_first: bool = False) -> tuple[int, int]:
    best = (1, 1)
    for h in range(1, max_tiles + 1):
        for w in range(1, max_tiles // h + 1):
            if block_h % h or block_w % w:
                continue
            key, best_key = (h * w, w), (best[0] * best[1], best[1])
            if widest_first:
                key, best_key = (w, h), (best[1], best[0])
            if key > best_key:
                best = (h, w)
    return best


def fused_batch_matmul_config(
    grid,
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    *,
    fused_activation=None,
    fp32_dest_acc_en: bool = False,
    cb_tile_budget: int = _L1_TILE_BUDGET,
    lofi_l1: bool = False,
):
    """2D mcast config with M split over grid rows and N over grid columns.

    ``lofi_l1``: L1-resident LoFi matmuls are unpack-bound, so they take a shorter out block
    (room for a wider in0 block) and the widest subblock (FF-up at one chunk: 143.7 -> 126.4 us).
    """
    import ttnn

    grid_x, grid_y = grid
    per_core_n = math.ceil(n_tiles / grid_x)
    per_core_m = math.ceil(m_tiles / grid_y)
    cols = math.ceil(n_tiles / per_core_n)
    rows = math.ceil(m_tiles / per_core_m)
    out_block_w = per_core_n
    out_block_h = _largest_divisor_at_most(per_core_m, _LOFI_L1_MAX_OUT_BLOCK_H if lofi_l1 else _MAX_OUT_BLOCK_H)
    in0_block_w = 1
    for cand in range(min(k_tiles, _MAX_IN0_BLOCK_W), 0, -1):
        if k_tiles % cand:
            continue
        tiles = 2 * out_block_h * cand + 2 * cand * out_block_w + out_block_h * out_block_w
        if tiles <= cb_tile_budget:
            in0_block_w = cand
            break
    sub_h, sub_w = _subblock(out_block_h, out_block_w, 4 if fp32_dest_acc_en else 8, widest_first=lofi_l1)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=True,
    )


def in1_mcast_1d_config(grid, m_tiles: int, k_tiles: int, n_tiles: int, *, fused_activation=None, lofi: bool = False):
    """1D config with M split over every core and all of N per core; the weights are multicast.

    For linears whose N leaves 2D grid columns idle (N = 768 on 11 columns uses 8): one L1
    chunk runs Wo / Wv.Wo in 58.4 vs 74.1 us and FF-down in 132.2 vs 165.6 us. Returns None
    when the circular buffers would not fit.
    """
    import ttnn

    grid_x, grid_y = grid
    per_core_m = math.ceil(m_tiles / (grid_x * grid_y))
    in0_block_w = _largest_divisor_at_most(k_tiles, _LOFI_1D_MAX_IN0_BLOCK_W if lofi else _1D_MAX_IN0_BLOCK_W)
    if 2 * per_core_m * in0_block_w + 2 * in0_block_w * n_tiles + per_core_m * n_tiles > _L1_TILE_BUDGET:
        return None
    sub_h, sub_w = _subblock(per_core_m, n_tiles, 4)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=per_core_m,
        out_block_w=n_tiles,
        per_core_M=per_core_m,
        per_core_N=n_tiles,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=False,
    )


_MAX_SDPA_CHUNK = 256


def sdpa_chunk_sizes(q_seq_padded: int, k_seq_padded: int) -> tuple[int, int]:
    """One chunk per sequence for short time attention (seq 160: 8.0 -> 2.3 ms vs 32/32)."""
    return min(q_seq_padded, _MAX_SDPA_CHUNK), min(k_seq_padded, _MAX_SDPA_CHUNK)


def _fused_activation(name: str | None):
    import ttnn

    if name is None:
        return None
    if name == "relu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    raise ValueError(f"unsupported fused activation {name!r}")


def linear(x, weight, *, bias=None, activation: str | None = None, memory_config=None, dtype=None, math_fidelity=None):
    """``ttnn.linear`` with a batch-folding program config when shapes are tile-aligned.

    Falls back to the auto-config for sub-tile geometries (the dummy test model).
    """
    import ttnn

    memory_config = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
    x_shape = tuple(x.padded_shape)
    w_shape = tuple(weight.padded_shape)
    k, n = w_shape[-2], w_shape[-1]
    # K must be logically tile-aligned so no tile padding enters the reduction;
    # a padded N only produces padding columns.
    aligned = (
        x.layout == ttnn.TILE_LAYOUT
        and not x.memory_config().is_sharded()
        and not memory_config.is_sharded()
        and x.shape[-1] % TILE == 0
        and weight.shape[-1] >= TILE
        and x_shape[-1] == k
    )
    if not aligned:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            activation=activation,
            memory_config=memory_config,
            dtype=dtype,
            compute_kernel_config=None if math_fidelity is None else compute_kernel_config(math_fidelity),
        )
    m_tiles, k_tiles, n_tiles = math.prod(x_shape[:-1]) // TILE, k // TILE, n // TILE
    grid = x.device().compute_with_storage_grid_size()
    l1_resident = ttnn.BufferType.L1 in (x.memory_config().buffer_type, memory_config.buffer_type)
    lofi = math_fidelity == ttnn.MathFidelity.LoFi
    program_config = None
    if l1_resident and math.ceil(n_tiles / math.ceil(n_tiles / grid.x)) < grid.x:
        program_config = in1_mcast_1d_config(
            (grid.x, grid.y), m_tiles, k_tiles, n_tiles, fused_activation=_fused_activation(activation), lofi=lofi
        )
    if program_config is None:
        program_config = fused_batch_matmul_config(
            (grid.x, grid.y),
            m_tiles,
            k_tiles,
            n_tiles,
            fused_activation=_fused_activation(activation),
            cb_tile_budget=_L1_RESIDENT_TILE_BUDGET if l1_resident else _L1_TILE_BUDGET,
            lofi_l1=l1_resident and lofi,
        )
    return ttnn.linear(
        x,
        weight,
        bias=bias,
        program_config=program_config,
        memory_config=memory_config,
        dtype=dtype,
        compute_kernel_config=compute_kernel_config(math_fidelity),
    )


def rms_norm(x, *, epsilon: float, memory_config=None, dtype=None):
    """Gamma-free RMSNorm; a ``dtype`` other than x's uses the model-local op, which can narrow the output."""
    import ttnn

    memory_config = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
    custom = (
        dtype is not None
        and dtype != x.dtype
        and x.layout == ttnn.TILE_LAYOUT
        and not x.memory_config().is_sharded()
        and not memory_config.is_sharded()
        and x.shape[-1] % TILE == 0
    )
    if not custom:
        return ttnn.rms_norm(x, epsilon=epsilon, memory_config=memory_config)
    from models.experimental.chronos_forecast import ops

    return ops.rms_norm(x, epsilon=epsilon, memory_config=memory_config, norm_dtype=dtype)
