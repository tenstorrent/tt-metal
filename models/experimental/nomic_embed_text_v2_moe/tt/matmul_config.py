# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program configs for the port's matmuls, derived from each call's shape.

B*S changes per call, so nothing here is a fixed table: every config is computed from the tile
counts of the call and the device grid. Left to itself ttnn picks in0_block_w = Kt / grid_x when
that divides and 1 otherwise, which on an 11-wide grid is 1 for every K in this model, and it
splits M per sequence rather than over the whole batch. Each matmul therefore passes its own.

The builders are cached on their arguments, compute kernel configs by identity, which holds since
each op group keeps one. A forward makes 54 builds: 380 to 510 us of Python uncached, 20 to 32 us
cached, which took a 2x37 forward from 5.3 to 4.8 ms.

The dense projections run through ttnn.experimental.minimal_matmul, which measured 11% to 16%
faster than the best 2D multicast config at every one of them at 8x512, and they write their
outputs to L1 when they fit. The experts (tt/experts.py) run grouped_matmul, which takes no
program config from here. minimal_matmul takes no batched second operand, so the router keeps
ttnn.matmul.

fc1 above SMALL_M_TILES, which a GELU follows, runs as a 2D multicast ttnn.matmul instead when the
GELU is fused into it (gelu_on_packer): that program applies it from the packer, where it partly
overlaps the matmul.
"""

from __future__ import annotations

from functools import cache

import ttnn

from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _largest_divisor_at_most(n: int, limit: int) -> int:
    return max(d for d in _divisors(n) if d <= limit)


def _subblock(block_h: int, block_w: int, budget: int, wide: bool = False) -> tuple[int, int]:
    """The largest output subblock that tiles the block and fits the destination register.

    Ties go to the taller subblock, or the wider one if `wide`: each config uses the one measured.
    """
    options = [(h, w) for h in _divisors(block_h) for w in _divisors(block_w) if h * w <= budget]
    return max(options, key=lambda hw: (hw[0] * hw[1], hw[1] if wide else hw[0]))


def dest_tiles(compute_kernel_config) -> int:
    """Tiles the destination register holds in half-sync mode: 8, or 4 with fp32 accumulation."""
    return 4 if compute_kernel_config.fp32_dest_acc_en else 8


# How each dense projection blocks its share of the output inside a core: the number of M blocks
# the core's rows split into, the width of an N block (None: the core's whole N share) and the K
# block. Measured at 8x512 on real weights and activations, where M > N and minimal_matmul puts
# M on the grid's 11 columns.
_DENSE_BLOCKING = {
    OpGroup.QKV: (1, 4, 4),
    OpGroup.ATTN_OUT: (1, None, 4),
    OpGroup.FC1: (1, 6, 6),
    OpGroup.FC2: (2, None, 4),
}

# With M <= N, M goes on the grid's 10 rows instead, and one blocking won for every case measured:
# fc1 at 8x384 (190 -> 140 us over the table above), QKV and fc1 at 4x512 (99 -> 80, 128 -> 99).
_DENSE_BLOCKING_M_ON_ROWS = (2, None, 4)


def minimal_matmul_footprint(
    config: ttnn.MinimalMatmulConfig,
    in0_dtype: ttnn.DataType = ttnn.bfloat16,
    in1_dtype: ttnn.DataType = ttnn.bfloat16,
    output_dtype: ttnn.DataType = ttnn.bfloat16,
    bias: bool = True,
) -> int:
    """Circular-buffer bytes per core of a minimal_matmul with fp32 partials.

    As its program factory allocates them: double-buffered input and output blocks, one partials
    block and one bf16 bias row. The defaults are the dense projections'.
    """
    m, k, n = config.M_block_size, config.K_block_size, config.N_block_size
    a, w, o, p = (ttnn.tile_size(dtype) for dtype in (in0_dtype, in1_dtype, output_dtype, ttnn.float32))
    return 2 * (m * k * a + k * n * w + m * n * o) + m * n * p + (n * ttnn.tile_size(ttnn.bfloat16) if bias else 0)


@cache
def dense_minimal_config(
    group: OpGroup,
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    grid: ttnn.CoreCoord,
    cb_bytes: int,
    compute_kernel_config,
    in0_dtype: ttnn.DataType = ttnn.bfloat16,
    in1_dtype: ttnn.DataType = ttnn.bfloat16,
    output_dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.MinimalMatmulConfig:
    """Blocking for one dense projection, (M, K) x (K, N) with M = B * S folded.

    minimal_matmul spreads the larger of M and N over the grid's 11 columns and the other over its
    10 rows, padding each to a multiple of its lanes, then blocks each core's share.

    A core's M share grows with B * S, so past the measured shapes its M splits into more blocks
    until the buffers fit cb_bytes. The first to split is fc1, above 17 tile rows or 5984 tokens.
    """
    m_on_x = m_tiles > n_tiles
    m_lanes, n_lanes = (grid.x, grid.y) if m_on_x else (grid.y, grid.x)
    m_share = ttnn.core.divup(m_tiles, m_lanes)
    n_share = ttnn.core.divup(n_tiles, n_lanes)

    m_splits, n_block, k_block = _DENSE_BLOCKING[group] if m_on_x else _DENSE_BLOCKING_M_ON_ROWS
    n_block = n_share if n_block is None else min(n_block, n_share)
    k_block = _largest_divisor_at_most(k_tiles, k_block)
    while True:
        m_block = ttnn.core.divup(m_share, m_splits)
        sub_h, sub_w = _subblock(m_block, n_block, dest_tiles(compute_kernel_config))
        config = ttnn.MinimalMatmulConfig(
            M_block_size=m_block,
            K_block_size=k_block,
            N_block_size=n_block,
            subblock_h=sub_h,
            subblock_w=sub_w,
            compute_with_storage_grid_size=grid,
        )
        if m_block == 1 or minimal_matmul_footprint(config, in0_dtype, in1_dtype, output_dtype) <= cb_bytes:
            return config
        m_splits += 1


# Up to this many tile rows of M = B * S the dense projections run as ttnn.linear with a
# multicast program config: minimal_matmul measured 1.1x to 2x slower there, at M of 1 to 32.
SMALL_M_TILES = 32

# Up to this many tile rows a 1D program that multicasts the activation from one core can beat
# the 2D split, when it spreads N over more cores; past it that one sender is the bound.
_ONE_D_M_TILES = 4


@cache
def dense_small_m_config(
    m_tiles: int, k_tiles: int, n_tiles: int, grid: ttnn.CoreCoord, compute_kernel_config
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """Program config for a dense projection with M at most SMALL_M_TILES tiles, batch folded.

    The 2D split puts M over the grid's rows and N over its columns. The 1D one (mcast_in0) gives
    every core all of M and N / cores columns, so at a narrow M it reaches up to N cores against
    the 2D split's M x 8 (N of 24 tiles) or M x 11; it is taken when it uses more cores. K runs in
    two blocks at K=768 and three at 3072, best or within 5% of best at every M measured.
    """
    k_block = _largest_divisor_at_most(k_tiles, min(32, k_tiles // 2))
    budget = dest_tiles(compute_kernel_config)
    per_core_m = ttnn.core.divup(m_tiles, grid.y)
    per_core_n = ttnn.core.divup(n_tiles, grid.x)
    cores_2d = ttnn.core.divup(m_tiles, per_core_m) * ttnn.core.divup(n_tiles, per_core_n)
    per_core_n_1d = ttnn.core.divup(n_tiles, grid.x * grid.y)
    if m_tiles <= _ONE_D_M_TILES and ttnn.core.divup(n_tiles, per_core_n_1d) > cores_2d:
        sub_h, sub_w = _subblock(m_tiles, per_core_n_1d, budget)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=k_block,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            out_block_h=m_tiles,
            out_block_w=per_core_n_1d,
            per_core_M=m_tiles,
            per_core_N=per_core_n_1d,
            fuse_batch=True,
            mcast_in0=True,
        )
    sub_h, sub_w = _subblock(per_core_m, per_core_n, budget)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=k_block,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fuse_batch=True,
    )


# The dense outputs written to L1 rather than DRAM. The output write was 46% to 70% of these
# matmuls' DRAM traffic, and moving it measured QKV -22%, out_proj -13% and fc1 -19% at 8x512.
# fc2's output is a sixth of its own traffic, 1% on fc2, but norm2 then reads it from L1: 0.78 ->
# 0.71 ms a forward at 8x512. At small M the consumers gain most, the head split above all, 17.1
# -> 13.2 us a call reading QKV from L1: 3.3% of the forward at 1x32, 1x128 and 2x37, 3.9% at
# 1x512.
_L1_OUTPUT_GROUPS = frozenset({OpGroup.QKV, OpGroup.ATTN_OUT, OpGroup.FC1, OpGroup.FC2})

# An L1 output takes its share of every bank away from the circular buffers of each op that runs
# while it is alive, and fc1's is joined by the GELU output of the same size. So it is taken only
# when it fits beside its own matmul's buffers and is at most a quarter of the budget, a margin
# that leaves half of L1 to the consumers' buffers. At 8x512 the four outputs take 4% to 16%.
_L1_OUTPUT_SHARE = 4


def gelu_activation(variant: ttnn.GeluVariant) -> ttnn.UnaryWithParam:
    """A GELU variant as a matmul's fused activation: the SFPU routine ttnn.gelu runs for it."""
    if variant == ttnn.GeluVariant.Tanh:
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
    return ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0 if variant == ttnn.GeluVariant.FastLut else 0.0)


def gelu_on_packer(variant: ttnn.GeluVariant) -> bool:
    """Whether a GELU fuses into a 2D multicast ttnn.matmul, rather than into minimal_matmul.

    The 2D program applies a fused activation from the packer (apply_activation_from_pack), beside
    the math thread computing the next subblock; minimal_matmul applies it on the math thread,
    where the tanh and accurate GELUs cost as much as their own op. On the packer the tanh GELU
    hides under the matmul in part: 260 us for fc1 and its GELU at 8x512 against 132 + 171 unfused,
    and 1630 against 786 + 1353 on the dense expert w1 the routed pass replaced, where the accurate
    one took 2227 against 786 + 1645. The LUT, a few instructions a row, stays on minimal_matmul:
    862 us over that w1 against 924. The routed w1 applies its GELU inside grouped_matmul.
    """
    return variant != ttnn.GeluVariant.FastLut


@cache
def dense_gelu_config(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    grid: ttnn.CoreCoord,
    variant: ttnn.GeluVariant,
    compute_kernel_config,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """2D multicast program for a dense projection with its GELU fused, above SMALL_M_TILES.

    M over the grid's rows and N over its columns, each core's N share in one block of up to 13
    rows, K in blocks of 4. Measured for fc1 at 8x512: 259.7 us, within 1% at K blocks of 2 or 3,
    and 274 to 317 us with N blocks of 3 tiles; 199.9 us at 8x384.
    """
    per_core_m = ttnn.core.divup(m_tiles, grid.y)
    per_core_n = ttnn.core.divup(n_tiles, grid.x)
    block_h = _largest_divisor_at_most(per_core_m, 13)
    sub_h, sub_w = _subblock(block_h, per_core_n, dest_tiles(compute_kernel_config), wide=True)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=_largest_divisor_at_most(k_tiles, 4),
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=block_h,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fuse_batch=True,
        fused_activation=gelu_activation(variant),
    )


def l1_bank_bytes(tiles: int, tt_config, dtype: ttnn.DataType | None = None) -> int:
    """The share of each L1 bank an interleaved tensor of `tiles` tiles takes, activation dtype by default."""
    return ttnn.core.divup(tiles, tt_config.l1_banks) * ttnn.tile_size(dtype or tt_config.activation_dtype)


def dense_linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    group: OpGroup,
    tt_config,
    gelu: ttnn.GeluVariant | None = None,
) -> ttnn.Tensor:
    """gelu(x @ weight + bias) for a (B, 1, S, K) activation, or the bare projection if gelu is None.

    M counts each sequence at its tile-padded length, as both programs fold the batch into M.
    The groups in _L1_OUTPUT_GROUPS return an L1-interleaved tensor when it fits (see
    _L1_OUTPUT_SHARE), and a DRAM one otherwise. Above SMALL_M_TILES an input already in L1 takes
    twice its share of the banks out of the budget the blocking is planned in: its producer ran out
    of place, and the buffer that producer freed sits above it, where no circular buffer can reach.

    Above SMALL_M_TILES a GELU is fused into the matmul: into a 2D multicast ttnn.linear when
    gelu_on_packer, into minimal_matmul otherwise. At or below it, it runs as its own op on the
    small-M program's output and keeps that output's placement.
    """
    # Indexed: unpacking a ttnn.Shape iterates it, 1.06 us of host time against 0.09 for four
    # index reads, and the model reads one about 110 times a forward.
    m_tiles = x.shape[0] * ttnn.core.divup(x.shape[-2], ttnn.TILE_SIZE)
    k_tiles = ttnn.core.divup(x.shape[-1], ttnn.TILE_SIZE)
    n_tiles = ttnn.core.divup(weight.shape[-1], ttnn.TILE_SIZE)
    compute_kernel_config = tt_config.compute_kernel_config(group)
    out_bank = l1_bank_bytes(m_tiles * n_tiles, tt_config)
    may_use_l1 = group in _L1_OUTPUT_GROUPS and out_bank <= tt_config.l1_cb_bytes // _L1_OUTPUT_SHARE
    if m_tiles <= SMALL_M_TILES:
        out = ttnn.linear(
            x,
            weight,
            bias=bias,
            program_config=dense_small_m_config(m_tiles, k_tiles, n_tiles, tt_config.core_grid, compute_kernel_config),
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG if may_use_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        if gelu is None:
            return out
        activated = ttnn.gelu(out, variant=gelu)
        ttnn.deallocate(out)
        return activated
    budget = tt_config.l1_cb_bytes
    if x.memory_config().buffer_type == ttnn.BufferType.L1:
        budget -= 2 * l1_bank_bytes(m_tiles * k_tiles, tt_config, x.dtype)
    if gelu is not None and gelu_on_packer(gelu):
        config = dense_gelu_config(m_tiles, k_tiles, n_tiles, tt_config.core_grid, gelu, compute_kernel_config)
        # The factory's buffers: _multicast_footprint's blocks and one row of bias tiles.
        footprint = _multicast_footprint(
            config.out_block_h,
            config.out_block_w,
            config.in0_block_w,
            x.dtype,
            weight.dtype,
            tt_config.activation_dtype,
        ) + config.out_block_w * ttnn.tile_size(bias.dtype)
        in_l1 = may_use_l1 and out_bank + footprint <= budget
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            program_config=config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG if in_l1 else ttnn.DRAM_MEMORY_CONFIG,
            dtype=tt_config.activation_dtype,
        )
    dtypes = (x.dtype, weight.dtype, tt_config.activation_dtype)
    config = dense_minimal_config(
        group, m_tiles, k_tiles, n_tiles, tt_config.core_grid, budget, compute_kernel_config, *dtypes
    )
    in_l1 = may_use_l1 and out_bank + minimal_matmul_footprint(config, *dtypes) <= budget
    return ttnn.experimental.minimal_matmul(
        x,
        weight,
        bias_tensor=bias,
        fused_activation=None if gelu is None else gelu_activation(gelu),
        config=config,
        memory_config=ttnn.L1_MEMORY_CONFIG if in_l1 else ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )


@cache
def router_program_config(
    tokens: int,
    k_tiles: int,
    n_tiles: int,
    activation_dtype: ttnn.DataType,
    weight_dtype: ttnn.DataType,
    output_dtype: ttnn.DataType,
    grid: ttnn.CoreCoord,
    cb_bytes: int,
    compute_kernel_config,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None:
    """(1, 1, T, 768) x (768, 32 n_tiles), bf16 into fp32 weights: the fewest tile rows per core.

    N is a tile or two. Up to grid.y tile rows of M, a 2D grid gives every output tile its own
    core: 5.4 against 8.4 us at 128 tokens on 8 cores instead of 4, bit-identical, since the K
    blocks are the same. Above, only M is spread; at T = 4096 that is two rows each on 64 cores.
    The router takes the whole batch in one call, so a core's rows grow with T: K runs in blocks
    of 8 tiles while the buffers fit, fewer above, and past that ttnn picks its own config (None).
    """
    m_tiles = ttnn.core.divup(tokens, ttnn.TILE_SIZE)
    if m_tiles <= grid.y and n_tiles <= grid.x:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(n_tiles, m_tiles),
            in0_block_w=_largest_divisor_at_most(k_tiles, 8),
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            transpose_mcast=False,
            fuse_batch=True,
        )
    per_core_m = ttnn.core.divup(m_tiles, grid.x * grid.y)
    k_block = max(
        (
            d
            for d in _divisors(k_tiles)
            if d <= 8
            and _multicast_footprint(per_core_m, n_tiles, d, activation_dtype, weight_dtype, output_dtype) <= cb_bytes
        ),
        default=None,
    )
    if k_block is None:
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=k_block,
        # A subblock as wide as the core's N share: narrower ones must be one tile tall.
        out_subblock_h=_largest_divisor_at_most(per_core_m, dest_tiles(compute_kernel_config) // n_tiles),
        out_subblock_w=n_tiles,
        out_block_h=per_core_m,
        out_block_w=n_tiles,
        per_core_M=per_core_m,
        per_core_N=n_tiles,
        fuse_batch=False,
        mcast_in0=False,
    )


def _multicast_footprint(
    rows: int,
    cols: int,
    k_block: int,
    activation: ttnn.DataType,
    weight: ttnn.DataType,
    output: ttnn.DataType,
) -> int:
    """Circular-buffer bytes per core of a multicast matmul with fp32 partials.

    Double-buffered activation and weight blocks, then one output and one partials block.
    """
    a, w, o, p = (ttnn.tile_size(dtype) for dtype in (activation, weight, output, ttnn.float32))
    return 2 * k_block * (rows * a + cols * w) + rows * cols * (o + p)
