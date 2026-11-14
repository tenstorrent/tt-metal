# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from helpers.llk_params import PerfRunType
from helpers.param_config import parametrize
from test_sfpu_reduce import (
    SFPU_REDUCE_PERF_SWEEP,
    _run_sfpu_reduce,
    _sfpu_reduce_config_kwargs,
)


@pytest.mark.perf
@parametrize(
    **SFPU_REDUCE_PERF_SWEEP,
    run_types=[[PerfRunType.MATH_ISOLATE]],
    loop_factor=list(range(10, 201, 10)),
    is_perf=[True],
)
def test_perf_sfpu_reduce(
    perf_report,
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    run_types,
    loop_factor,
    is_perf,
):
    input_dimensions = [32, 32]
    tile_count = 1

    _run_sfpu_reduce(
        _sfpu_reduce_config_kwargs(
            formats,
            dest_acc,
            mathop,
            reduce_pool,
            input_dimensions,
            tile_count,
            src_A=None,
            src_B=None,
            is_perf=is_perf,
            loop_factor=loop_factor,
        ),
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
    )
