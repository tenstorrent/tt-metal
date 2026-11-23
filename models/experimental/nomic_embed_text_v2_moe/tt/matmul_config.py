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

From 32 tile rows (1024 tokens) the dense projections run on the model's two matmul ops only. QKV with M <= N
runs as grouped_matmul_w1 with one expert (grouped_projection): 24.1 against 28.5 us for the small-M program at
32 tile rows and 41.7 against 55.2 for minimal_matmul at 64. The bare projections with M > N (QKV, out_proj, fc2)
run as ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul where its buffers fit: minimal_matmul's compute kernel
fed by multicast, bit for bit minimal_matmul's output at the same grid and blocks, 12% to 24% faster at 8x512. It
takes minimal_matmul's K block but for QKV and out_proj up to 132 tile rows (_PIROTSKI_K_BLOCK).

fc1 above SMALL_M_TILES, which a GELU follows, runs as a 2D multicast ttnn.matmul instead when the
GELU is fused into it (gelu_on_packer): that program applies it from the packer, where it partly
overlaps the matmul.
"""

from __future__ import annotations

from functools import cache
from typing import Iterable, NamedTuple, Sequence

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


# How each dense projection blocks its share of the output inside a core where M > N, and
# minimal_matmul puts M on the grid's 11 columns: the most rows of an M block (None: no limit), the
# core's M share splitting evenly into the fewest blocks within it; the width of an N block (None: the
# core's whole N share); and the K block. Swept over every legal blocking at M of 64 to 256 tile rows
# on the forward's own inputs, the limits fastest over the calls of 15 workloads of 1024 to 16384
# tokens were 6 rows for out_proj (66.7 against 77.4 us at 192 tile rows), 12 for fc2 (61.3 against
# 80.5 us at 64, the share in one block) and 12 for QKV (133.7 against 141.8 us at 192). Each keeps
# its K block, and with it its accumulation order, so the outputs are unchanged.
_DENSE_BLOCKING = {
    OpGroup.QKV: (12, 4, 4),
    OpGroup.ATTN_OUT: (6, None, 4),
    OpGroup.FC1: (None, 6, 6),
    OpGroup.FC2: (12, None, 4),
}

# A group whose core's M share is one block of the table above takes this N block and the first of these K
# blocks whose buffers fit instead. QKV's whole N share of 8 tiles in one block, K in blocks of 3: 7.1% less at
# 128 tile rows (93.4 against 100.5 us) and 2% to 6% at 132; with its M share split, the same blocks measured
# 13% slower (150.6 against 133.7 us at 192 tile rows). In K blocks of 2 its buffers take 944 KB against 1024
# for 92.5 against 93.6 us at 128, nearly bit for bit the same: at 8x512, 1 of the output's 9.4 M elements differs
# from K blocks of 3, so the K block is part of the arithmetic. out_proj's K in blocks of 2: 9.3% less at 64 tile
# rows (27.8 against 30.6 us).
_DENSE_ONE_M_BLOCK = {OpGroup.QKV: (None, (3, 2)), OpGroup.ATTN_OUT: (None, (2,))}

# Outside the case above, a group here takes this K block at the table's M blocking where its buffers fit, the
# table's own where they do not. QKV in K blocks of 8: 3.7% less at 192 tile rows (129.8 against 134.7 us, its M share
# in two blocks of 9 rows); beside its output in L1 they do not fit at 252 and 256, where K blocks of 6
# measured the same as 4 within the noise. It changes the accumulation order, which the accuracy gates cover.
_DENSE_WIDE_K = {OpGroup.QKV: 8}

# With M <= N, M goes on the grid's 10 rows instead, and one blocking won for every case measured:
# fc1 at 8x384 (190 -> 140 us over the table above), QKV and fc1 at 4x512 (99 -> 80, 128 -> 99).
# Here the first number is the count of M blocks.
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


def pirotski_matmul_footprint(
    config: ttnn.MinimalMatmulConfig,
    k_tiles: int,
    n_tiles_per_core: int,
    resident_weights: bool,
    direct_pack: bool,
    in0_dtype: ttnn.DataType = ttnn.bfloat16,
    in1_dtype: ttnn.DataType = ttnn.bfloat16,
    output_dtype: ttnn.DataType = ttnn.bfloat16,
    bias: bool = True,
) -> int:
    """Circular-buffer bytes per core of a pirotski_matmul with fp32 partials, as its program factory allocates them.

    Two input blocks; two weight blocks, or a core's whole K x N share with resident weights; one output block, none
    when it packs into the output's own shard; one partials block and one bf16 bias row.
    """
    m, k, n = config.M_block_size, config.K_block_size, config.N_block_size
    a, w, o, p = (ttnn.tile_size(dtype) for dtype in (in0_dtype, in1_dtype, output_dtype, ttnn.float32))
    weight_tiles = k_tiles * n_tiles_per_core if resident_weights else 2 * k * n
    output = 0 if direct_pack else m * n * o
    return 2 * m * k * a + weight_tiles * w + output + m * n * p + (n * ttnn.tile_size(ttnn.bfloat16) if bias else 0)


def grouped_matmul_footprint(
    k_tiles: int,
    n_tiles: int,
    grid: ttnn.CoreCoord,
    input_dtype: ttnn.DataType,
    weight_dtype: ttnn.DataType,
    output_dtype: ttnn.DataType,
    num_experts: int,
    bias: bool = False,
    rows: int = 0,
) -> int:
    """Circular-buffer bytes per core of a grouped_matmul's compute cores, as its program factory allocates them.

    The input ring of K tiles a slot; the weight ring, two slots (one with one expert) of K tiles, plus the bias
    row, of a core column's share of N; the output ring of that share's tiles a slot; a ones tile with a bias. Two
    input and two output slots, except for one expert (a TILE output), which runs subblock-outer over the blocks
    it holds: a lane's every block when it has at most 4 of the `rows` (at least 2 slots), else groups of 4 in
    rings of 8. Every tile tile-aligned for DRAM, which bfloat16 and bfloat8_b tiles already are. The schedule and
    the two counts pages, under 1 KB, are left to the callers' margins; a w2 call's dedicated input senders, east of
    its compute cores, hold less.
    """
    column_n_tiles = ttnn.core.divup(n_tiles, grid.x)
    input_tile, weight_tile = ttnn.tile_size(input_dtype), ttnn.tile_size(weight_dtype)
    weight_slots = 2 if num_experts > 1 else 1
    input_slots = output_slots = 2
    if num_experts == 1:
        lane_blocks = ttnn.core.divup(rows // ttnn.TILE_SIZE, grid.y)
        if lane_blocks <= 4:
            input_slots = max(2, lane_blocks)
        else:
            input_slots = output_slots = 8
    total = input_slots * k_tiles * input_tile
    total += weight_slots * (k_tiles + (1 if bias else 0)) * column_n_tiles * weight_tile
    total += output_slots * column_n_tiles * ttnn.tile_size(output_dtype)
    return total + (input_tile if bias else 0)


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

    The grid keeps only the lanes that get work. N = 24 tiles in shares of 3 fills 8 of the 10 rows,
    and the factory still sends the 2 idle rows every input block: without them out_proj ran 2% faster
    and fc2 and QKV (9 of 10 rows) 1%, with the same shares and so the same bits.
    """
    m_on_x = m_tiles > n_tiles
    m_lanes, n_lanes = (grid.x, grid.y) if m_on_x else (grid.y, grid.x)
    m_share = ttnn.core.divup(m_tiles, m_lanes)
    n_share = ttnn.core.divup(n_tiles, n_lanes)
    m_used, n_used = ttnn.core.divup(m_tiles, m_share), ttnn.core.divup(n_tiles, n_share)
    grid = ttnn.CoreCoord(m_used, n_used) if m_on_x else ttnn.CoreCoord(n_used, m_used)

    if m_on_x:
        max_m_block, n_block, k_block = _DENSE_BLOCKING[group]
        m_splits = 1 if max_m_block is None else ttnn.core.divup(m_share, max_m_block)
    else:
        m_splits, n_block, k_block = _DENSE_BLOCKING_M_ON_ROWS
    n_block = n_share if n_block is None else min(n_block, n_share)
    split = n_block, _largest_divisor_at_most(k_tiles, k_block)
    # The (N block, K block) pairs tried at this many M blocks, in order; past them M splits further.
    attempts = [split]
    if m_on_x and m_splits == 1 and group in _DENSE_ONE_M_BLOCK:
        one_n, one_ks = _DENSE_ONE_M_BLOCK[group]
        one_n = n_share if one_n is None else min(one_n, n_share)
        attempts = [(one_n, _largest_divisor_at_most(k_tiles, one_k)) for one_k in one_ks]
    elif m_on_x and group in _DENSE_WIDE_K:
        attempts = [(n_block, _largest_divisor_at_most(k_tiles, _DENSE_WIDE_K[group])), split]
    while True:
        m_block = ttnn.core.divup(m_share, m_splits)
        for n_block, k_block in attempts:
            # With M on the columns, equal-area ties go to the wider subblock: out_proj's 1 x 3 ran 82.7 us
            # against 3 x 1's 89.5 at 252 tile rows and fc2's 158.0 against 168.1 at 192, within 2% elsewhere.
            sub_h, sub_w = _subblock(m_block, n_block, dest_tiles(compute_kernel_config), wide=True)
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
        attempts = [split]
        m_splits += 1


# Groups whose pirotski_matmul keeps a core's whole weight share in L1 where it fits, so that only the first M block
# streams it. QKV above 132 tile rows, two N blocks a core at minimal_matmul's M block: 5% to 9% less than streamed
# (116.1 against 122.5 us at 192 tile rows, 146.8 against 161.3 at 256). With one N block a core its M share splits
# so that the last block, whose output drains after the K loop, is small: as measured per share in
# _PIROTSKI_FIRST_M_BLOCK (8 rows as 5 + 3: 50.7 us, 9 as 6 + 3: 56.0, 11 as 7 + 4: 66.8, 12 as 5 + 5 + 2: 71.2
# against 72.2 as 8 + 4), in blocks of two thirds of it otherwise. Against minimal_matmul's one block a core at 128
# tile rows: 71.2 against 92.2 us.
_PIROTSKI_RESIDENT = frozenset({OpGroup.QKV})
_PIROTSKI_FIRST_M_BLOCK = {8: 5, 9: 6, 11: 7, 12: 5}

# The K block of a group's pirotski_matmul where a core's M share is at most _PIROTSKI_K_BLOCK_MAX_M_SHARE tile rows,
# in place of minimal_matmul's. QKV at 128 and 132 tile rows, M blocks of 5: 71.2 against 73.8 us in K blocks of 2
# (72.3 at their best M block, 8); at 80 to 112 minimal_matmul's plan already takes 3. out_proj 0.2 to 0.8 us less
# than in its plan's K blocks of 2 or 4 at 48 to 128 tile rows (37.4 against 37.8 at 128, 16.7 against 17.5 at 48).
# Beyond that share K blocks of 3 measured no faster for either (out_proj 74.9 against 74.2 us at 256 tile rows), and
# fc2 is fastest in its plan's K blocks of 4 (94.4 against 94.7 at 128).
_PIROTSKI_K_BLOCK = {OpGroup.QKV: 3, OpGroup.ATTN_OUT: 3}
_PIROTSKI_K_BLOCK_MAX_M_SHARE = 12
# At a core M share of up to _PIROTSKI_SMALL_K_BLOCK_MAX_M_SHARE tile rows (32 and 33 tile rows of M) a larger K block
# halves the stream's round trips per byte: out_proj 14.6 -> 12.3 us in blocks of 6 at 32 tile rows (12.6 at 8, 13.5
# at 12), fc2 42.9 -> 37.3 in blocks of 12 (37.8 at 8, 37.3 at 16, 38.0 at 24). At a share of 6 (64 tile rows) the
# blocks above stay best: out_proj 19.9 at 3 against 21.0 at 6, fc2 48.5 at 4 against 49.0 at 8.
_PIROTSKI_SMALL_K_BLOCK = {OpGroup.ATTN_OUT: 6, OpGroup.FC2: 12}
_PIROTSKI_SMALL_K_BLOCK_MAX_M_SHARE = 3
# Groups whose weight readers read one K block ahead of the multicast (read_ahead) at those shares: fc2 37.1 -> 31.9 us
# at 32 tile rows, where its weight reads cost 10.4 us of the call (37.1 against 26.8 with none); 31.7 to 31.8 for K
# blocks of 6 to 32 with it. out_proj 12.4 either way; at a share of 6 both ran 0.5 to 0.7 us slower with it.
_PIROTSKI_READ_AHEAD = frozenset({OpGroup.FC2})

# Groups whose pirotski_matmul gives the last rows of each core column's M share to the device grid's rows below its
# grid (strip_m_tiles), from this M share. A strip row computes its rows for the N of `lanes` grid rows, so one of its
# rows costs about lanes of the grid's; strip_m is the count with the least max(M share - strip_m, lanes * strip_m +
# 1), the larger on a tie, and the grid's remaining rows go in the fewest even M blocks of at most
# _PIROTSKI_STRIP_MAX_M_BLOCK (one block when packed): fc2's 12 rows at 144 tile rows in one block took 98.6 us
# against 101.6 as 6 + 6, out_proj's 16 at 216 as 8 + 8 54.6 against 56.0 as 12 + 4. Each strip row also reads its N's weights once a call, about
# 26 us for out_proj and 78 for fc2, so the strip pays only from these shares. Against no strip: out_proj 28.6 ->
# 26.6 us at 96 tile rows, 37.4 -> 31.7 at 128, 39.7 -> 35.8 at 132, 51.6 -> 44.8 at 168, 69.7 -> 59.4 at 240, 71.6
# -> 62.4 at 252; fc2 87.4 -> 81.3 at 112, 94.5 -> 82.5 at 128, 99.2 -> 85.7 at 132, 127.7 -> 108.4 at 168, 176.2 ->
# 145.2 at 240, 190.2 -> 154.0 at 252; both slower below these shares. The strip's weight reader overlaps its reads
# with its multicast for fc2, and for out_proj with a one-row strip, where they bound it (fc2 94.3 -> 82.5 us at 128;
# out_proj 28.7 -> 26.6 at 96, but 31.7 -> 33.1 at 128).
_PIROTSKI_STRIP_MIN_M_SHARE = {OpGroup.ATTN_OUT: 9, OpGroup.FC2: 11}
_PIROTSKI_STRIP_MAX_M_BLOCK = {OpGroup.ATTN_OUT: 10, OpGroup.FC2: 12}


class PirotskiPlan(NamedTuple):
    config: ttnn.MinimalMatmulConfig
    resident_weights: bool
    strip_m_tiles: int = 0
    strip_read_ahead: bool = False
    read_ahead: bool = False


@cache
def dense_pirotski_plan(
    group: OpGroup,
    minimal: ttnn.MinimalMatmulConfig,
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    cb_bytes: int,
    compute_kernel_config,
    in0_dtype: ttnn.DataType,
    in1_dtype: ttnn.DataType,
    output_dtype: ttnn.DataType,
    norm_shards: tuple[int, int, int] | None = None,
    device_rows: int = 0,
) -> PirotskiPlan | None:
    """pirotski_matmul's arguments in place of minimal_matmul's config, with M > N; None if it cannot take the call
    or its buffers do not fit cb_bytes.

    It keeps minimal's grid and N block, and its K block unless _PIROTSKI_K_BLOCK sets one; with them it keeps
    minimal_matmul's accumulation order (every second N block of a core runs K backwards), so the output is
    minimal_matmul's at that K block bit for bit; the M block may differ.
    norm_shards, (core columns, tile rows a core, core rows) of the split layer norm's layout when the output is
    written in it, must match the grid and a core's M share: compute then packs into the output's shard, which
    takes no circular buffer. device_rows, the device grid's, sets how many grid rows below the call's grid a strip
    (_PIROTSKI_STRIP_MIN_M_SHARE) has; its rows keep their K blocks and order too.
    """
    grid = minimal.compute_with_storage_grid_size
    m_share, n_share = ttnn.core.divup(m_tiles, grid.x), n_tiles // grid.y
    n_block = minimal.N_block_size
    if n_tiles % grid.y or n_share % n_block:
        return None
    m_block, resident = minimal.M_block_size, group in _PIROTSKI_RESIDENT
    k_block = _PIROTSKI_K_BLOCK.get(group) if m_share <= _PIROTSKI_K_BLOCK_MAX_M_SHARE else None
    if m_share <= _PIROTSKI_SMALL_K_BLOCK_MAX_M_SHARE:
        k_block = _PIROTSKI_SMALL_K_BLOCK.get(group, k_block)
    if k_block is None or k_tiles % k_block:
        k_block = minimal.K_block_size
    # The op packs into the output only on these same conditions (pirotski_matmul's compute_layout); the plan's
    # footprint assumes it does.
    direct_pack = norm_shards is not None
    if direct_pack:
        if norm_shards != (grid.x, m_share, grid.y) or n_block != n_share:
            return None
        # Nothing drains a packed output, so a core's M share goes in one block: out_proj at 72, 96 and 112 tile
        # rows took 23.8, 28.9 and 35.0 us against 32.2, 38.1 and 44.9 for minimal_matmul, whose M blocks do not
        # divide the share there; 37.8 against 38.9 in blocks of 6 at 128.
        m_block = m_share
    elif resident and n_block == n_share:
        m_block = _PIROTSKI_FIRST_M_BLOCK.get(m_share, round(2 * m_share / 3))
    # (M block, resident, strip rows) in order of preference. Where a core's whole weight share does not fit beside
    # the M block, it may beside one of a third of the M share: QKV at 256 tile rows, 153.0 us in blocks of 8 against
    # 161.3 streamed in minimal_matmul's 12.
    attempts = [(m_block, resident, 0)]
    if resident and not direct_pack and n_block != n_share:
        attempts.append((ttnn.core.divup(m_share, 3), True, 0))
    attempts.append((m_block, False, 0))
    strip_rows = device_rows - grid.y
    if (
        m_share >= _PIROTSKI_STRIP_MIN_M_SHARE.get(group, m_share + 1)
        and not resident
        and n_block == n_share
        and strip_rows > 0
        and grid.y % strip_rows == 0
    ):
        lanes = grid.y // strip_rows
        strip_m = min(range(m_share - 1, 0, -1), key=lambda rows: max(m_share - rows, lanes * rows + 1))
        main_m = m_share - strip_m
        main_block = ttnn.core.divup(main_m, ttnn.core.divup(main_m, _PIROTSKI_STRIP_MAX_M_BLOCK[group]))
        attempts.insert(0, (main_m if direct_pack else main_block, False, strip_m))
    dtypes = (in0_dtype, in1_dtype, output_dtype)
    for m_block, keep_weights, strip_m in attempts:
        sub_h, sub_w = _subblock(m_block, n_block, dest_tiles(compute_kernel_config), wide=True)
        config = ttnn.MinimalMatmulConfig(
            M_block_size=m_block,
            K_block_size=k_block,
            N_block_size=n_block,
            subblock_h=sub_h,
            subblock_w=sub_w,
            compute_with_storage_grid_size=grid,
        )
        footprint = pirotski_matmul_footprint(config, k_tiles, n_share, keep_weights, direct_pack, *dtypes)
        if strip_m:
            # A strip core: the same buffers for one strip_m x lanes N-share block, streamed, never packed.
            strip_config = ttnn.MinimalMatmulConfig(
                M_block_size=strip_m,
                K_block_size=k_block,
                N_block_size=n_share * (grid.y // strip_rows),
                subblock_h=1,
                subblock_w=1,
                compute_with_storage_grid_size=grid,
            )
            footprint = max(footprint, pirotski_matmul_footprint(strip_config, k_tiles, 0, False, False, *dtypes))
        if footprint <= cb_bytes:
            strip_read_ahead = strip_m > 0 and (group is OpGroup.FC2 or strip_m == 1)
            # The kernel reads ahead only with one M and one N block a core and streamed weights.
            read_ahead = (
                group in _PIROTSKI_READ_AHEAD
                and m_share <= _PIROTSKI_SMALL_K_BLOCK_MAX_M_SHARE
                and not strip_m
                and not keep_weights
                and m_block >= m_share
                and n_block == n_share
            )
            return PirotskiPlan(config, keep_weights, strip_m, strip_read_ahead, read_ahead)
    return None


# Up to this many tile rows of M = B * S, under 1024 tokens, the dense projections run as ttnn.linear with a
# multicast program config: minimal_matmul measured 1.1x to 2x slower there, at M of 1 to 32. From 32 they take
# the model's own matmul ops.
SMALL_M_TILES = 31

# Up to this many tile rows a 1D program that multicasts the activation from one core can beat
# the 2D split, when it spreads N over more cores; past it that one sender is the bound.
_ONE_D_M_TILES = 4

# A projection that a split layer norm reads writes its output in the norm's layout (dense_norm_layout_config)
# where a core of that layout holds at least this many tile rows: 3 at 23 to 31 tile rows of M. At 32 out_proj
# saved 2.2 us a call net; at 24 (out of scope) it ran 2.6 us longer and saved 2.3 us of conversion.
_NORM_LAYOUT_MIN_ROWS = 3


@cache
def dense_small_m_config(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    grid: ttnn.CoreCoord,
    compute_kernel_config,
    gelu: ttnn.GeluVariant | None = None,
    fast_gelu: bool = False,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """Program config for a dense projection with M at most SMALL_M_TILES tiles, batch folded.

    The 2D split puts M over the grid's rows and N over its columns. The 1D one (mcast_in0) gives
    every core all of M and N / cores columns, so at a narrow M it reaches up to N cores against
    the 2D split's M x 8 (N of 24 tiles) or M x 11; it is taken when it uses more cores. K runs in
    two blocks at K=768 and three at 3072, except on a 2D core with at least 12 output tiles: there in
    six, which at 32 tile rows measured 8% to 9% less for QKV (28.7 against 31.6 us), 2% to 4% for
    out_proj and 6% for fc2, and 6% for QKV at 16. A smaller block pays the extra K steps on too few
    MACs: out_proj's 3 tiles a core at 8 tile rows ran 14% slower in six, its 6 at 16 rows 2% to 3%.
    A changed accumulation order, which the accuracy gates cover.
    gelu, fc1's, is fused and applied from the packer by either program (gelu_activation).
    """
    fused_activation = None if gelu is None else gelu_activation(gelu, fast_gelu)
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
            fused_activation=fused_activation,
            mcast_in0=True,
        )
    sub_h, sub_w = _subblock(per_core_m, per_core_n, budget)
    if per_core_m * per_core_n >= 12:
        k_block = _largest_divisor_at_most(k_tiles, ttnn.core.divup(k_tiles, 6))
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
        fused_activation=fused_activation,
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


def gelu_activation(variant: ttnn.GeluVariant, fast: bool = False) -> ttnn.UnaryWithParam:
    """A GELU variant as a matmul's fused activation: the SFPU routine ttnn.gelu runs for it, or with
    fast the tanh form computed as x / (1 + exp(-2u)), GELU_TANH param 1, within 1 bfloat16 ULP of it."""
    if variant == ttnn.GeluVariant.Tanh:
        return (
            ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH, 1.0)
            if fast
            else ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
        )
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
    fast: bool = False,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """2D multicast program for a dense projection with its GELU fused, above SMALL_M_TILES.

    M over the grid's rows and N over its columns, each core's N share in one block of up to 13
    rows, K in blocks of 4. Measured for fc1 at 8x512: 259.7 us, within 1% at K blocks of 2 or 3,
    and 274 to 317 us with N blocks of 3 tiles; 199.9 us at 8x384.

    A core reruns the whole K loop for every block of rows, so its rows are rounded up, by at most
    three, to a count with a large divisor: 23 rows a core (14x512) ran as 23 blocks of one, fc1 1.9x
    slower a token than at 12x512 or 16x512.
    """
    base_m = ttnn.core.divup(m_tiles, grid.y)
    per_core_m = min(
        range(base_m, base_m + 4),
        key=lambda rows: rows * (1 + 1 / _largest_divisor_at_most(rows, 13)),
    )
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
        fused_activation=gelu_activation(variant, fast),
    )


def l1_depth(tensor: ttnn.Tensor, tt_config) -> int:
    """Bytes a bank from the top of L1 down to the start of an L1 tensor: what a circular buffer must stay
    clear of while the tensor lives, the tensor, whatever sits above it and the holes between. 0 in DRAM.

    L1 tensors are allocated from the top down and circular buffers from the bottom up, so an op's
    buffers may take all of L1 below its lowest live input. Read from the tensor's address, which costs
    nothing; listing every buffer on the device to find the lowest took 1.8 ms.
    """
    if tensor.memory_config().buffer_type != ttnn.BufferType.L1:
        return 0
    return tt_config.l1_top - tensor.buffer_address()


# The model's long-lived L1 tensors sit at the top of L1: the rotary tables (4.7 KB a bank each at 2048
# tokens a sequence, and a pair cut to the pass's length), the rotation matrix (2 KB) and the norm weights,
# 13 KB a bank measured at 100x81. A pass's activations are allocated below them, so the depth of any of
# those covers them; a plan whose tensors are all in DRAM keeps clear of this much instead.
_LONG_LIVED_L1_BYTES = 32 << 10


def l1_floor(tensors: Iterable[ttnn.Tensor], tt_config) -> int:
    """Bytes a bank from the top of L1 that circular buffers must stay clear of while the tensors live:
    the deepest one's l1_depth, and at least _LONG_LIVED_L1_BYTES."""
    return max(_LONG_LIVED_L1_BYTES, max((l1_depth(tensor, tt_config) for tensor in tensors), default=0))


def l1_bank_bytes(tiles: int, tt_config, dtype: ttnn.DataType | None = None) -> int:
    """The share of each L1 bank an interleaved tensor of `tiles` tiles takes, activation dtype by default."""
    return ttnn.core.divup(tiles, tt_config.l1_banks) * ttnn.tile_size(dtype or tt_config.activation_dtype)


# grouped_matmul_w1 reads a dense weight as a chunked weight in runs of this many bfloat16 tiles (30 KB): 15 was
# faster than 30 for fc1 at 2x512 (30.3 against 32.0 us) and even at 8x512.
GROUPED_CHUNK_TILES = 15

# A grouped projection's output goes to L1 when each bank's share is at most this and it leaves the op's circular
# buffers (grouped_matmul_footprint) room below it and the tensors live beside it, with _GROUPED_L1_MARGIN to spare.
_GROUPED_L1_BANK_BYTES = 256 << 10
_GROUPED_L1_MARGIN = 16 << 10


def prepare_grouped_projection(weight: ttnn.Tensor, bias: ttnn.Tensor, device) -> ttnn.Tensor:
    """A dense projection's (K, N) TILE weight and (N,) bias as grouped_matmul_w1's chunked weight of one expert."""
    k, n = weight.shape[-2], weight.shape[-1]
    return ttnn.experimental.nomic_embed_text_v2_moe.prepare_grouped_matmul_w1_weight(
        ttnn.reshape(weight, (1, 1, k, n)),
        1,
        bias=ttnn.reshape(bias, (1, 1, 1, n)),
        chunk_tiles=GROUPED_CHUNK_TILES,
        device=device,
    )


def grouped_projection(
    x: ttnn.Tensor,
    chunked: ttnn.Tensor,
    weight: ttnn.Tensor,
    group: OpGroup,
    tt_config,
    live: Sequence[ttnn.Tensor] = (),
    activation: ttnn.UnaryWithParam | None = None,
    dtype: ttnn.DataType | None = None,
) -> ttnn.Tensor:
    """activation(x @ weight + bias) through grouped_matmul_w1 with one expert, every tile row of x its row.

    chunked is prepare_grouped_projection's copy of weight (read for its shape and dtype) with the bias, which the
    op adds as one more K step. It takes (B, 1, S, K) in any layout as it stands, so the batch is not flattened.
    The output, in dtype (the activations' by default), goes to L1 when it fits below the tensors live beside x.
    """
    dtype = dtype or tt_config.activation_dtype
    k_tiles, n_tiles = weight.shape[-2] // ttnn.TILE_SIZE, weight.shape[-1] // ttnn.TILE_SIZE
    rows = x.shape[0] * ttnn.core.divup(x.shape[-2], ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    bank = l1_bank_bytes(rows // ttnn.TILE_SIZE * n_tiles, tt_config, dtype)
    footprint = grouped_matmul_footprint(
        k_tiles, n_tiles, tt_config.core_grid, x.dtype, weight.dtype, dtype, num_experts=1, bias=True, rows=rows
    )
    in_l1 = bank <= _GROUPED_L1_BANK_BYTES
    in_l1 = in_l1 and l1_floor((x, *live), tt_config) + bank + footprint + _GROUPED_L1_MARGIN <= tt_config.l1_cb_bytes
    return ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1(
        x,
        chunked,
        None,
        dtype=dtype,
        activation=activation,
        memory_config=ttnn.L1_MEMORY_CONFIG if in_l1 else ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=tt_config.compute_kernel_config(group),
        weight_chunk_tiles=GROUPED_CHUNK_TILES,
        weight_n=n_tiles * ttnn.TILE_SIZE,
        weight_has_bias=True,
    )


def dense_linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    group: OpGroup,
    tt_config,
    gelu: ttnn.GeluVariant | None = None,
    live: Sequence[ttnn.Tensor] = (),
    norm_layout: tuple[ttnn.MemoryConfig, int, int, int] | None = None,
    grouped: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """gelu(x @ weight + bias) for a (B, 1, S, K) activation, or the bare projection if gelu is None.

    grouped, prepare_grouped_projection's copy of weight and bias, runs a bare projection with M <= N above
    SMALL_M_TILES as grouped_projection.

    M counts each sequence at its tile-padded length, as both programs fold the batch into M.
    The groups in _L1_OUTPUT_GROUPS return an L1-interleaved tensor when it fits (see
    _L1_OUTPUT_SHARE), and a DRAM one otherwise. Above SMALL_M_TILES the blocking is planned below
    l1_floor of x and live, the other tensors the caller holds meanwhile: no circular buffer can reach
    them, the buffers freed above them or any other tensor between. There the placements are tried in order
    (norm layout, L1, DRAM) and the first that fits runs: a bare projection with M > N as pirotski_matmul where
    its plan (dense_pirotski_plan) fits there, as minimal_matmul otherwise.

    A GELU is fused into the matmul: above SMALL_M_TILES into a 2D multicast ttnn.linear when
    gelu_on_packer and into minimal_matmul otherwise, at or below it into the small-M program, all of
    which apply it from the packer but minimal_matmul. tt_config.dense_fast_gelu picks the tanh GELU's
    fast form (gelu_activation).

    norm_layout, (memory config, core columns, tile rows a core, core rows) of the split layer norm that reads
    the bare projection (tt.common.sharded_norm_layout), has the output written in that layout: from
    _NORM_LAYOUT_MIN_ROWS tile rows a core up to SMALL_M_TILES, and above it wherever it fits L1. minimal_matmul
    writes it at no cost, 47.7 against 48.0 us for out_proj and 107.5 against 107.8 for fc2 at 128 tile rows.
    """
    # Indexed: unpacking a ttnn.Shape iterates it, 1.06 us of host time against 0.09 for four
    # index reads, and the model reads one about 110 times a forward.
    m_tiles = x.shape[0] * ttnn.core.divup(x.shape[-2], ttnn.TILE_SIZE)
    k_tiles = ttnn.core.divup(x.shape[-1], ttnn.TILE_SIZE)
    n_tiles = ttnn.core.divup(weight.shape[-1], ttnn.TILE_SIZE)
    compute_kernel_config = tt_config.compute_kernel_config(group)
    # fc1's output, the GELU'd intermediate fc2 reads, may be narrower than the activations.
    output_dtype = tt_config.dense_intermediate_dtype if group is OpGroup.FC1 else tt_config.activation_dtype
    out_bank = l1_bank_bytes(m_tiles * n_tiles, tt_config, output_dtype)
    may_use_l1 = group in _L1_OUTPUT_GROUPS and out_bank <= tt_config.l1_cb_bytes // _L1_OUTPUT_SHARE
    fast_gelu = tt_config.dense_fast_gelu
    if m_tiles <= SMALL_M_TILES:
        program_config = dense_small_m_config(
            m_tiles, k_tiles, n_tiles, tt_config.core_grid, compute_kernel_config, gelu, fast_gelu
        )
        memory_config = ttnn.L1_MEMORY_CONFIG if may_use_l1 else ttnn.DRAM_MEMORY_CONFIG
        if norm_layout is not None and gelu is None and norm_layout[2] >= _NORM_LAYOUT_MIN_ROWS:
            memory_config, core_columns, block_h, core_rows = norm_layout
            program_config = dense_norm_layout_config(
                k_tiles, n_tiles, core_columns, block_h, core_rows, compute_kernel_config
            )
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=memory_config,
            dtype=output_dtype,
        )
    if grouped is not None and gelu is None and m_tiles <= n_tiles:
        return grouped_projection(x, grouped, weight, group, tt_config, live)
    budget = tt_config.l1_cb_bytes - l1_floor((x, *live), tt_config)
    if gelu is not None and gelu_on_packer(gelu):
        config = dense_gelu_config(
            m_tiles, k_tiles, n_tiles, tt_config.core_grid, gelu, compute_kernel_config, fast_gelu
        )
        # The factory's buffers: _multicast_footprint's blocks and one row of bias tiles.
        footprint = _multicast_footprint(
            config.out_block_h,
            config.out_block_w,
            config.in0_block_w,
            x.dtype,
            weight.dtype,
            output_dtype,
        ) + config.out_block_w * ttnn.tile_size(bias.dtype)
        in_l1 = may_use_l1 and out_bank + footprint <= budget
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            program_config=config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG if in_l1 else ttnn.DRAM_MEMORY_CONFIG,
            dtype=output_dtype,
        )
    dtypes = (x.dtype, weight.dtype, output_dtype)
    # (memory config, bytes a core, the norm layout's shards) of each placement, in order of preference.
    placements = []
    if norm_layout is not None and gelu is None:
        layout, core_columns, block_h, core_rows = norm_layout
        bank = block_h * (n_tiles // core_rows) * ttnn.tile_size(output_dtype)
        placements.append((layout, bank, (core_columns, block_h, core_rows)))
    if may_use_l1:
        placements.append((ttnn.L1_MEMORY_CONFIG, out_bank, None))
    placements.append((ttnn.DRAM_MEMORY_CONFIG, 0, None))
    for memory_config, bank, norm_shards in placements:
        # Planned against the budget less the output first, so that the blocks shrink to leave the output
        # room in L1 rather than send it to DRAM.
        config = dense_minimal_config(
            group, m_tiles, k_tiles, n_tiles, tt_config.core_grid, budget - bank, compute_kernel_config, *dtypes
        )
        plan = None
        if gelu is None and m_tiles > n_tiles:
            # With M <= N minimal_matmul is faster: QKV at 64 tile rows took 87.7 us on pirotski_matmul, 64.6 with
            # resident weights, against 56.7, bound by its weight reads.
            plan = dense_pirotski_plan(
                group,
                config,
                m_tiles,
                k_tiles,
                n_tiles,
                budget - bank,
                compute_kernel_config,
                *dtypes,
                norm_shards,
                tt_config.core_grid.y,
            )
        if plan is not None:
            return ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul(
                x,
                weight,
                config=plan.config,
                dtype=output_dtype,
                bias=bias,
                memory_config=memory_config,
                compute_kernel_config=compute_kernel_config,
                resident_weights=plan.resident_weights,
                strip_m_tiles=plan.strip_m_tiles,
                strip_read_ahead=plan.strip_read_ahead,
                read_ahead=plan.read_ahead,
            )
        # DRAM's plan always fits: its M splits until it does.
        if not bank or bank + minimal_matmul_footprint(config, *dtypes) <= budget:
            break
    return ttnn.experimental.minimal_matmul(
        x,
        weight,
        bias_tensor=bias,
        fused_activation=None if gelu is None else gelu_activation(gelu, fast_gelu),
        config=config,
        memory_config=memory_config,
        # Explicit: the output dtype otherwise follows the input's, and fc2 reads a bfloat8_b
        # intermediate when dense_intermediate_dtype says so.
        dtype=output_dtype,
        compute_kernel_config=compute_kernel_config,
    )


@cache
def dense_norm_layout_config(
    k_tiles: int, n_tiles: int, core_columns: int, block_h: int, core_rows: int, compute_kernel_config
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """2D program writing a projection's output in the split layer norm's block-sharded layout.

    That layout (tt.common.sharded_norm_configs) puts M on core_columns of the grid's columns, block_h tile
    rows a core, and N on core_rows rows; the norm then reads the output as it stands rather than converting
    it. At 32 tile rows out_proj ran 0.2 us more than the interleaved program (17.6 against 17.4) and saved
    the norm's 2.4 to 2.6 us conversion. K in twelve blocks: in eight the call took 17.9 us, in six 18.6.
    """
    per_core_n = n_tiles // core_rows
    sub_h, sub_w = _subblock(block_h, per_core_n, dest_tiles(compute_kernel_config), wide=True)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_columns, core_rows),
        in0_block_w=_largest_divisor_at_most(k_tiles, ttnn.core.divup(k_tiles, 12)),
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=block_h,
        out_block_w=per_core_n,
        per_core_M=block_h,
        per_core_N=per_core_n,
        transpose_mcast=True,
        fuse_batch=True,
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
