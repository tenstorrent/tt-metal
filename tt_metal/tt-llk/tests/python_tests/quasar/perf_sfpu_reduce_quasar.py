# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    ReducePool,
)
from helpers.param_config import parametrize
from quasar.test_sfpu_reduce_quasar import (
    MAX_TILES,
    get_supported_reduce_axes,
)
from quasar.test_sfpu_reduce_quasar import (
    test_sfpu_reduce_quasar as run_sfpu_reduce_quasar,
)

# Dest-full tall and wide, in elements.
#
# Not generate_perf_input_dimensions(): that sizes to the Dest capacity for the format's dest_acc,
# which is 8 tiles at 16-bit. The functional sweep caps every format at MAX_TILES=4 instead, since
# a row reduce needs its whole block resident and the 32-bit ceiling of 4 binds first. Deriving
# from the same cap keeps the perf shapes to ones the functional generator actually emits.
PERF_INPUT_DIMENSIONS_REDUCE = [
    [MAX_TILES * TILE_DIM, TILE_DIM],  # tall: 4 tile rows x 1 tile column
    [TILE_DIM, MAX_TILES * TILE_DIM],  # wide: 1 tile row x 4 tile columns
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


def _perf_axes(reduce_pool, formats):
    """Reduce axes to time for this pool/format pair.

    Defers to the functional gate so the perf sweep cannot ask for a combination the kernel
    static_asserts away - integer row AVG being the one that matters.
    """
    return get_supported_reduce_axes(reduce_pool, formats)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    reduce_pool=PERF_POOLS,
    formats=PERF_FORMATS,
    mathop=_perf_axes,
    dimension_combinations=PERF_INPUT_DIMENSIONS_REDUCE,
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
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
