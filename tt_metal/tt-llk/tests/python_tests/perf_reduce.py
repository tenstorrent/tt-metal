# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from helpers.reduce_perf_matrix import (
    CORE_FORMATS,
    CORE_REDUCE_DIMS,
    CORE_TILE_DIMENSIONS,
    PERF_POOL_TYPES,
    SCALAR_REPRESENTATIVE,
    WHBH_EXTENSION_FORMATS,
    WHBH_EXTENSION_TILE_DIMENSIONS,
    perf_input_dimensions,
    perf_math_fidelities,
)
from test_reduce import REDUCE_SWEEP, _run_reduce_test, dest_acc_modes_for_format

PERF_LOOP_FACTOR = 64
CORE_TILE_DIMENSION_LISTS = [list(td) for td in CORE_TILE_DIMENSIONS]
PERF_TILE_DIMENSIONS = CORE_TILE_DIMENSION_LISTS + [
    list(td) for td in WHBH_EXTENSION_TILE_DIMENSIONS
]

REDUCE_PERF_SWEEP = {
    **REDUCE_SWEEP,
    "tile_dimensions": PERF_TILE_DIMENSIONS,
    "formats": CORE_FORMATS + WHBH_EXTENSION_FORMATS,
    # Perf skips the golden check, so it keeps the Float16 dest_acc=Yes rows the functional sweep drops.
    "dest_acc": dest_acc_modes_for_format,
    "reduce_dim": CORE_REDUCE_DIMS,
    "pool_type": PERF_POOL_TYPES,
    "math_fidelity": lambda pool_type: perf_math_fidelities(pool_type),
    "is_reduce_to_one": [False],
}

REDUCE_SCALAR_PERF_SWEEP = {
    **SCALAR_REPRESENTATIVE,
    "tile_dimensions": CORE_TILE_DIMENSION_LISTS,
    "is_reduce_to_one": [False],
}

PERF_MEASUREMENT = dict(
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)


def _run_reduce_perf(
    perf_report,
    tile_dimensions,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    run_types,
    loop_factor,
    is_perf,
):
    _run_reduce_test(
        formats,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        tile_dimensions,
        input_dimensions=perf_input_dimensions(tile_dimensions),
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(**REDUCE_PERF_SWEEP, **PERF_MEASUREMENT)
def test_perf_reduce(
    perf_report,
    tile_dimensions,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    run_types,
    loop_factor,
    is_perf,
):
    _run_reduce_perf(
        perf_report,
        tile_dimensions,
        formats,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        run_types,
        loop_factor,
        is_perf,
    )


@pytest.mark.perf
@parametrize(**REDUCE_SCALAR_PERF_SWEEP, **PERF_MEASUREMENT)
def test_perf_reduce_scalar(
    perf_report,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    tile_dimensions,
    is_reduce_to_one,
    run_types,
    loop_factor,
    is_perf,
):
    _run_reduce_perf(
        perf_report,
        tile_dimensions,
        formats,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        run_types,
        loop_factor,
        is_perf,
    )
