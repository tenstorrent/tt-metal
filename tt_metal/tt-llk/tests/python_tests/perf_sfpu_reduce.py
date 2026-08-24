# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    DestSync,
    MathOperation,
    PerfRunType,
    ReducePool,
)
from helpers.param_config import (
    generate_perf_input_dimensions,
    input_output_formats,
    parametrize,
)
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
    input_dimensions=lambda dest_acc: generate_perf_input_dimensions(
        dest_acc, DestSync.Half
    ),
    loop_factor=[32],
)
def test_perf_sfpu_reduce(
    perf_report,
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    input_dimensions,
    loop_factor,
):
    tile_count = (input_dimensions[0] * input_dimensions[1]) // 1024

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
