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
    formats=input_output_formats(
        [DataFormat.Float32],
        same=True,
    ),
    dest_acc=[DestAccumulation.Yes],
    mathop=[MathOperation.ReduceRow],
    reduce_pool=[ReducePool.Max],
    loop_factor=list(range(10, 201, 10)),
)
def test_perf_sfpu_reduce_row_max(
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


# Perf vehicle for the corr-only corpus row metal__ckernel_sfpu_reduce
# (Lane BK measurement-gap close): the module above measures the Float32
# ReduceRow/Max branch (perform_reduce_row_max_min), while the row's audited
# functional nodes gate the Int32 branches (perform_reduce_row_max_min_int32 /
# calculate_reduce_max_min_int32_col). This node mirrors the corr branch —
# Int32 -> Int32, dest_acc=Yes, ReduceRow/Max — at a single loop_factor point
# so the sweep's OFF/ON legs measure the same instruction stream the
# correctness gate verifies.
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    dest_acc=[DestAccumulation.Yes],
    mathop=[MathOperation.ReduceRow],
    reduce_pool=[ReducePool.Max],
    loop_factor=[100],
)
def test_perf_sfpu_reduce_row_max_int32(
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
