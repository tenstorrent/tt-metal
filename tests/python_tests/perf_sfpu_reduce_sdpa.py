# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathOperation, PerfRunType, ReducePool
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.profiler import ProfilerConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    MATH_OP,
    REDUCE_POOL_TYPE,
    TILE_COUNT,
)


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b],  # Only Float16_b is supported for SDPA reduce
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
    mathop=[MathOperation.ReduceColumn],
    reduce_pool=[ReducePool.Max],  # Only MAX is supported for SDPA reduce
    loop_factor=list(
        range(10, 201, 10)
    ),  # Multiple loop factors to minimize profiler overhead
)
def test_perf_sfpu_reduce_sdpa(
    perf_report,
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    loop_factor,
    workers_tensix_coordinates,
):
    """
    Performance test for SFPU reduce SDPA operation.

    This test specifically measures the performance of the SFPU reduce operation
    used in SDPA (Scaled Dot-Product Attention) implementations. It focuses on
    measuring cycles spent in the SFPU calculations, not including memory operations.

    The test uses a 128x32 input dimension (4 tiles) and performs column-wise
    max reduction, which is the typical operation in SDPA softmax computation.
    """

    input_dimensions = [128, 64]
    tile_count = input_dimensions[1] // 32 * input_dimensions[0] // 32

    # Run performance benchmarks focusing on MATH_ISOLATE to measure SFPU cycles
    # MATH_ISOLATE measures only the math operation cycles, excluding unpack/pack
    # This specifically measures the _calculate_reduce_sdpa_ function cycles
    configuration = ProfilerConfig(
        "sources/sfpu_reduce_sdpa_perf.cpp",
        formats,
        run_types=[
            # PerfRunType.L1_TO_L1,         # Full operation timing
            PerfRunType.MATH_ISOLATE,  # Only SFPU computation cycles (_calculate_reduce_sdpa_)
            # PerfRunType.UNPACK_ISOLATE,   # Unpack timing for reference
            # PerfRunType.PACK_ISOLATE,     # Pack timing for reference
        ],
        templates=[
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            REDUCE_POOL_TYPE(reduce_pool),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(loop_factor),  # Used to minimize profiler overhead
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

    configuration.run(perf_report, location=workers_tensix_coordinates)
