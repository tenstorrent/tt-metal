# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance sweep of the generic SFPU reduce (sources/sfpu_reduce_row_max_perf.cpp, the kernel behind
sfpu_reduce / calculate_reduce) over every configuration ttnn dispatches to it: pool {SUM, AVG, MAX, MIN},
dimension {ROW, COL}, Float16_b in a 16-bit and in a 32-bit dest, Float32 and Int32 (32-bit inputs are
unpacked straight into dest, as ttnn does), one and four tiles per call, and the run types MATH_ISOLATE
(the SFPU body alone), L1_TO_L1 (unpack, copy into dest, reduce, pack) and PACK_ISOLATE.

How the harness column reads. The kernel's MATH_ISOLATE loop issues one calculate_reduce over the whole
block per call and TILE_CNT calls per loop iteration, and the report divides the TILE_LOOP window by
loop_factor x tile_cnt, so for a multi-tile block the MATH_ISOLATE column is cycles per call (one call
reduces tile_cnt tiles); divide by tile_cnt for cycles per tile. In L1_TO_L1 the kernel copies tile_cnt
tiles, reduces the block once and packs tile_cnt tiles per iteration, so that column is per tile.

Which tile counts are swept. A row reduce over four tiles (block_ct_dim 4) combines the per-tile row
results across the tiles; a column MAX or MIN over four tiles (block_rt_dim 4) is the block-height path of
calculate_reduce_max_min. The column SUM and AVG kernels reduce one tile per call whatever the block
dimensions, and the Int32 column MAX and MIN kernel is a single-tile kernel, so those run at one tile only.
Row AVG exists for float formats only (see calculate_reduce).

Two loop factors: 16, the value the other SFPU perf modules use, and 128, where the per-iteration zone
entry and exit is amortised (the 10 to 200 sweep this module used to run spans about 3 percent between its
ends and buys nothing else).
"""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    PerfRunType,
    ReducePool,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    LOOP_FACTOR,
    MATH_OP,
    TILE_COUNT,
    generate_input_dim,
)

TILE_DIM = 32


def get_dest_acc_modes(formats):
    """Float16_b runs in both dest widths; the 32-bit formats need a 32-bit dest."""
    if formats.input_format == DataFormat.Float16_b:
        return [DestAccumulation.No, DestAccumulation.Yes]
    return [DestAccumulation.Yes]


def get_reduce_pools(formats, mathop):
    """Every pool of the kernel, minus the integer row AVG it does not implement."""
    pools = [ReducePool.Sum, ReducePool.Average, ReducePool.Max, ReducePool.Min]
    if mathop == MathOperation.ReduceRow and formats.input_format.is_integer():
        pools.remove(ReducePool.Average)
    return pools


def get_tile_counts(formats, mathop, reduce_pool):
    """One tile always; four tiles where the kernel reduces a block per call."""
    if mathop == MathOperation.ReduceRow:
        return [1, 4]
    if reduce_pool in (ReducePool.Max, ReducePool.Min) and not formats.input_format.is_integer():
        return [1, 4]
    return [1]


@pytest.mark.perf
@parametrize(
    # Float32 and Int32 are the formats ttnn routes to the SFPU row reduce (Int32 always, Float32 in accurate fp32
    # mode); Float16_b covers the bf16 MIN and the 16-bit-dest paths.
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32, DataFormat.Int32],
        same=True,
    ),
    dest_acc=get_dest_acc_modes,
    mathop=[MathOperation.ReduceRow, MathOperation.ReduceColumn],
    reduce_pool=get_reduce_pools,
    tile_count=get_tile_counts,
    loop_factor=[16, 128],
)
def test_perf_sfpu_reduce(
    perf_report, formats, dest_acc, mathop, reduce_pool, tile_count, loop_factor
):
    # A row reduce spans tile_count tiles along the row (block_ct_dim), a column reduce stacks them
    # (block_rt_dim); generate_input_dim derives both block dimensions from the shape.
    if mathop == MathOperation.ReduceRow:
        input_dimensions = [TILE_DIM, TILE_DIM * tile_count]
    else:
        input_dimensions = [TILE_DIM * tile_count, TILE_DIM]

    configuration = PerfConfig(
        "sources/sfpu_reduce_row_max_perf.cpp",
        formats,
        run_types=[
            PerfRunType.MATH_ISOLATE,
            PerfRunType.L1_TO_L1,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[
            MATH_OP(mathop=mathop, pool_type=reduce_pool),
            APPROX_MODE(ApproximationMode.No),
            generate_input_dim(input_dimensions, input_dimensions),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(loop_factor),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        # 32-bit inputs cannot go through SrcA and are unpacked into dest, as ttnn does for them.
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        disable_format_inference=True,
        compile_time_formats=True,
    )

    configuration.run(perf_report)
