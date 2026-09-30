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
    loop_factor=[16, 128],  # as perf_sfpu_reduce.py: the plain value and the amortised one
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
    Performance test for the SFPU column MAX reduce over a block of tiles, the shape SDPA softmax uses.

    The kernel (sources/sfpu_reduce_sdpa_perf.cpp) runs the generic calculate_reduce<MAX, REDUCE_COL,
    Float16_b> over a block height of four tiles on the math thread; it is the block-height path of
    calculate_reduce_max_min, not the 4x2 sub-block reduce of sfpu_reduce_sdpa_test.cpp and not the
    pack-thread issue the SDPA kernels use. The input is one column of four 32x32 tiles (128x32).

    MATH_ISOLATE issues one four-tile reduce per tile count per loop iteration and the report divides the
    window by loop_factor x tile_cnt, so its column is cycles per four-tile call (divide by four for cycles
    per tile). L1_TO_L1 unpacks and copies the four tiles into dest, reduces the block once and packs the
    four tiles, so its column is per tile.

    The test runs on every architecture the kernel compiles for; the earlier Blackhole skip had no
    recorded reason and the module passes there.
    """

    input_dimensions = [4 * TILE_DIM, TILE_DIM]
    tile_count = (input_dimensions[0] // TILE_DIM) * (input_dimensions[1] // TILE_DIM)

    configuration = PerfConfig(
        "sources/sfpu_reduce_sdpa_perf.cpp",
        formats,
        run_types=[
            PerfRunType.MATH_ISOLATE,  # the SFPU body over the four-tile block
            PerfRunType.L1_TO_L1,  # unpack, copy into dest, reduce, pack
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
