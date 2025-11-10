# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    MathOperation,
    ReducePool,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.perf import (
    PerfRunType,
    perf_benchmark,
    update_report,
)


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    test_name="sfpu_reduce_sdpa_perf",
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
    test_name,
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    loop_factor,
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

    test_config = {
        "testname": test_name,
        "tile_cnt": tile_count,
        "formats": formats,
        "dest_acc": dest_acc,
        "pool_type": reduce_pool,
        "mathop": mathop,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "unpack_to_dest": False,  # Must be False since math kernel does A2D copy
        "loop_factor": loop_factor,  # Used to minimize profiler overhead
    }

    # Run performance benchmarks focusing on MATH_ISOLATE to measure SFPU cycles
    # MATH_ISOLATE measures only the math operation cycles, excluding unpack/pack
    # This specifically measures the _calculate_reduce_sdpa_ function cycles
    results = perf_benchmark(
        test_config,
        [
            # PerfRunType.L1_TO_L1,      # Full operation timing
            PerfRunType.MATH_ISOLATE,  # Only SFPU computation cycles (_calculate_reduce_sdpa_)
            # PerfRunType.UNPACK_ISOLATE, # Unpack timing for reference
            # PerfRunType.PACK_ISOLATE,   # Pack timing for reference
        ],
    )

    update_report(perf_report, test_config, results)
