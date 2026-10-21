# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity, ReducePool
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_reduce import REDUCE_SWEEP, _run_reduce_test

PERF_TILE_COUNT = 16
PERF_LOOP_FACTOR = 64


def perf_math_fidelities(pool_type):
    # GMPOOL ignores fidelity; LoFi and HiFi4 bracket the fidelity phase count.
    if pool_type == ReducePool.Max:
        return [MathFidelity.LoFi]
    return [MathFidelity.LoFi, MathFidelity.HiFi4]


def perf_input_dimensions(tile_dimensions, tile_count=PERF_TILE_COUNT):
    return [tile_dimensions[0] * tile_count, tile_dimensions[1]]


REDUCE_PERF_SWEEP = {
    **REDUCE_SWEEP,
    "tile_dimensions": [[32, 32]],
    "formats": input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ]
    ),
    "math_fidelity": perf_math_fidelities,
    "is_reduce_to_one": [False],
}


@pytest.mark.perf
@parametrize(
    **REDUCE_PERF_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
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
