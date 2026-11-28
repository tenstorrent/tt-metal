# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import PerfRunType
from helpers.param_config import parametrize
from test_sfpu_reduce import (
    SFPU_REDUCE_PERF_SWEEP,
    _run_sfpu_reduce,
    _sfpu_reduce_config_kwargs,
    use_int32_twos_complement,
)


@pytest.mark.perf
@parametrize(
    **SFPU_REDUCE_PERF_SWEEP,
    # The WH/BH SFPU reduce perf measures a single 32x32 tile.
    input_dimensions=[[32, 32]],
    run_types=[[PerfRunType.MATH_ISOLATE]],
    loop_factor=list(range(10, 201, 10)),
    is_perf=[True],
)
def test_perf_sfpu_reduce(
    perf_report,
    formats,
    dest_acc,
    mathop,
    pool_type,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    tile_count = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    _run_sfpu_reduce(
        _sfpu_reduce_config_kwargs(
            formats,
            dest_acc,
            mathop,
            pool_type,
            input_dimensions,
            tile_count,
            src_A=None,
            src_B=None,
            is_perf=is_perf,
            loop_factor=loop_factor,
            twos_complement=use_int32_twos_complement(formats, pool_type, mathop),
        ),
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
    )
