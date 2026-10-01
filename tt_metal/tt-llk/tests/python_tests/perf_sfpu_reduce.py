# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance sweep of the generic SFPU reduce (sources/sfpu_reduce_row_max_perf.cpp) over the configurations ttnn
dispatches to it. MATH_ISOLATE reads as cycles per call (one call reduces tile_cnt tiles), L1_TO_L1 as cycles per tile.
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
    # A row reduce spans tile_count tiles along the row, a column reduce stacks them.
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
