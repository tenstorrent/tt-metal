# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    ReducePool,
)
from helpers.param_config import generate_perf_input_dimensions, parametrize
from quasar.test_sfpu_reduce_quasar import (
    get_supported_reduce_axes,
)
from quasar.test_sfpu_reduce_quasar import (
    test_sfpu_reduce_quasar as run_sfpu_reduce_quasar,
)

# The kernel is written for 32x32 tiles only, so the functional sweep has no tile-size axis and
# select_perf_tile_sizes() would return just (32, 32); there is nothing to sweep there.


def perf_input_dimensions(formats):
    """Dest-full tall and wide blocks for this format's Dest width, the 2x2 square and one tile.

    generate_perf_input_dimensions() gives (max_tiles, 1) and (1, max_tiles) in tiles: 8 at
    16-bit, 4 at 32-bit, the same ceiling the functional sweep uses per format. The square is the
    one extra row-axis case: both factors of row_base = rt * block_ct_dim * REDUCE_TILE_STRIDE
    exceed one there. The single tile is the like-for-like point against Blackhole, whose
    perf_sfpu_reduce runs only a 32x32 Float32 ReduceRow Max. The column axis reduces tile by tile,
    so its rows across these shapes are knowingly duplicates.
    """
    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit()
        else DestAccumulation.No
    )
    return generate_perf_input_dimensions(dest_acc, DestSync.Half) + [
        [2 * TILE_DIM, 2 * TILE_DIM],
        [TILE_DIM, TILE_DIM],
    ]


# One format per instruction path that costs something different:
#   Float32    - float fold, 32-bit Dest
#   Float16_b  - float fold, 16-bit Dest (narrower store)
#   Int32      - integer fold (SFPIADD / two's-complement SFPSWAP), 32-bit Dest
# Float16 is omitted: it shares Float16_b's 16-bit Dest path and the same SFPADD fold, so it only
# duplicates timings.
PERF_FORMATS = [
    InputOutputFormat(fmt, fmt)
    for fmt in (DataFormat.Float32, DataFormat.Float16_b, DataFormat.Int32)
]

# Every pool, because each folds differently: MAX/MIN compare with SFPSWAP, SUM adds, and AVG
# adds then divides - and the integer divide is much dearer than the float one.
PERF_POOLS = [ReducePool.Sum, ReducePool.Average, ReducePool.Max, ReducePool.Min]

# Stimuli values do not affect timing, so the functional bounds axis collapses to one range.
# Small enough to be in range for every format swept here.
PERF_INPUT_BOUNDS = (-100, 100)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    reduce_pool=PERF_POOLS,
    formats=PERF_FORMATS,
    mathop=get_supported_reduce_axes,
    dimension_combinations=perf_input_dimensions,
    implied_math_format=[ImpliedMathFormat.Yes],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_sfpu_reduce_quasar(
    perf_report,
    reduce_pool,
    formats,
    mathop,
    dimension_combinations,
    implied_math_format,
    run_types,
    loop_factor,
    is_perf,
):
    """Time the Quasar SFPU reduce through the shared correctness harness.

    `reduced_extent` is pinned to the full tile: the sub-tile padding the functional sweep
    exercises changes the stimulus, not the instruction stream, so it would only duplicate rows.
    """
    run_sfpu_reduce_quasar(
        formats,
        mathop,
        reduce_pool,
        PERF_INPUT_BOUNDS,
        dimension_combinations,
        TILE_DIM,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
