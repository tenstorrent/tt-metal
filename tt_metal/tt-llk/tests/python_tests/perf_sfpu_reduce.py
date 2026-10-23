# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


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


@pytest.mark.perf
@parametrize(
    # Float32 and Int32 are the formats ttnn routes to the SFPU row reduce (Int32 always, Float32 with
    # accurate fp32 mode). Float32 MAX runs the compare-and-swap horizontal_reduce_max (MIN differs only
    # by the SFPCONFIG direction bit) and SUM the add-based horizontal_reduce, for both formats. Int32 MAX
    # takes the separate signed compare-and-swap path (horizontal_reduce_max_int32), which shares the
    # MOV-free rotate butterfly with the float/unsigned body but not its compare-and-swap.
    formats=input_output_formats(
        [DataFormat.Float32, DataFormat.Int32],
        same=True,
    ),
    dest_acc=[DestAccumulation.Yes],
    mathop=[MathOperation.ReduceRow],
    reduce_pool=[ReducePool.Max, ReducePool.Sum],
    loop_factor=list(range(10, 201, 10)),
)
def test_perf_sfpu_reduce(
    perf_report, formats, dest_acc, mathop, reduce_pool, loop_factor
):
    input_dimensions = [32, 32]
    tile_count = 1

    configuration = PerfConfig(
        "sources/sfpu_reduce_row_max_perf.cpp",
        formats,
        run_types=[
            PerfRunType.MATH_ISOLATE,
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
        unpack_to_dest=True,
        dest_acc=dest_acc,
        disable_format_inference=True,
        compile_time_formats=True,
    )

    configuration.run(perf_report)
