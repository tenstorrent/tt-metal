# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathOperation, PerfRunType, ReducePool
from helpers.param_config import (
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
    generate_input_dim,
)

TILE_DIM = 32


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b],  # Only Float16_b is supported for SDPA reduce
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
    mathop=[MathOperation.ReduceColumn],
    reduce_pool=[ReducePool.Max],  # Only MAX is supported for SDPA reduce
    loop_factor=[16, 128],  # as perf_sfpu_reduce.py
)
def test_perf_sfpu_reduce_sdpa(
    perf_report,
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    loop_factor,
):
    """
    Performance test for the SFPU column MAX reduce over a block of four tiles, the shape SDPA softmax uses.
    MATH_ISOLATE reads as cycles per four-tile call (divide by four for cycles per tile), L1_TO_L1 as cycles per tile.
    """

    input_dimensions = [4 * TILE_DIM, TILE_DIM]
    tile_count = (input_dimensions[0] // TILE_DIM) * (input_dimensions[1] // TILE_DIM)

    configuration = PerfConfig(
        "sources/sfpu_reduce_sdpa_perf.cpp",
        formats,
        run_types=[
            PerfRunType.MATH_ISOLATE,
            PerfRunType.L1_TO_L1,
        ],
        templates=[
            MATH_OP(mathop=mathop),
            REDUCE_POOL_TYPE(reduce_pool),
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
        unpack_to_dest=False,  # Must be False since math kernel does A2D copy
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
