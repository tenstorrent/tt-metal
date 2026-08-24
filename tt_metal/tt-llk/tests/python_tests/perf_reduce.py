# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestSync,
    MathOperation,
    PerfRunType,
    ReduceDimension,
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
    LOOP_FACTOR,
    MATH_OP,
    REDUCE_POOL_TYPE,
    TILE_COUNT,
)

REDUCE_MATHOP = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=[ReducePool.Max, ReducePool.Average, ReducePool.Sum],
    input_dimensions=lambda dest_acc: generate_perf_input_dimensions(
        dest_acc, DestSync.Half
    ),
)
def test_perf_reduce(
    perf_report,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    input_dimensions,
):
    tile_count = (input_dimensions[0] * input_dimensions[1]) // 1024
    configuration = PerfConfig(
        "sources/reduce_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=REDUCE_MATHOP[reduce_dim]),
            REDUCE_POOL_TYPE(pool_type),
        ],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(64)],
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
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
