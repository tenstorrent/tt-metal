# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from helpers.reduce_perf_matrix import (
    BLOCK_LENGTH_REPRESENTATIVE,
    BLOCK_LENGTH_TILE_COUNTS,
    CORE_FORMATS,
    CORE_REDUCE_DIMS,
    CORE_TILE_DIMENSIONS,
    PERF_POOL_TYPES,
    PERF_TILE_COUNT,
    SCALAR_REPRESENTATIVE,
    WHBH_EXTENSION_FORMATS,
    WHBH_EXTENSION_TILE_DIMENSIONS,
    perf_input_dimensions,
    perf_math_fidelities,
    reduce_to_one_modes,
)
from test_reduce import (
    _reduce_to_one_for_format,
    _run_reduce_test,
    reduce_dest_acc_modes,
)

PERF_LOOP_FACTOR = 64
CORE_TILE_DIMENSION_LISTS = [list(td) for td in CORE_TILE_DIMENSIONS]
PERF_TILE_DIMENSIONS = CORE_TILE_DIMENSION_LISTS + [
    list(td) for td in WHBH_EXTENSION_TILE_DIMENSIONS
]


def perf_dest_acc_modes(formats):
    # Float32 -> Float16_b also runs with a 16-bit Dest on WH/BH; that row pairs with Quasar.
    if (
        formats.input_format == DataFormat.Float32
        and formats.output_format == DataFormat.Float16_b
    ):
        return [DestAccumulation.No, DestAccumulation.Yes]
    return reduce_dest_acc_modes(formats)


REDUCE_PERF_SWEEP = {
    "tile_dimensions": PERF_TILE_DIMENSIONS,
    "formats": CORE_FORMATS + WHBH_EXTENSION_FORMATS,
    "dest_acc": perf_dest_acc_modes,
    "reduce_dim": CORE_REDUCE_DIMS,
    "pool_type": PERF_POOL_TYPES,
    "math_fidelity": lambda pool_type: perf_math_fidelities(pool_type),
    "is_reduce_to_one": lambda formats, reduce_dim, pool_type: [
        mode
        for mode in reduce_to_one_modes(reduce_dim, pool_type)
        if mode in _reduce_to_one_for_format(formats)
    ],
}

REDUCE_BLOCK_LENGTH_PERF_SWEEP = {
    **BLOCK_LENGTH_REPRESENTATIVE,
    "math_fidelity": [MathFidelity.HiFi4],
    "tile_dimensions": CORE_TILE_DIMENSION_LISTS,
    "is_reduce_to_one": [False],
    "tile_count": BLOCK_LENGTH_TILE_COUNTS,
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
    tile_count=PERF_TILE_COUNT,
):
    _run_reduce_test(
        formats,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        tile_dimensions,
        input_dimensions=perf_input_dimensions(tile_dimensions, tile_count),
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


@pytest.mark.perf
@parametrize(**REDUCE_BLOCK_LENGTH_PERF_SWEEP, **PERF_MEASUREMENT)
def test_perf_reduce_block_length(
    perf_report,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    tile_dimensions,
    is_reduce_to_one,
    tile_count,
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
        tile_count=tile_count,
    )
