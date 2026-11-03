# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from helpers.llk_params import DestAccumulation, PerfRunType, ReducePool
from helpers.param_config import parametrize
from test_sfpu_reduce import (
    _run_sfpu_reduce,
    _sfpu_reduce_config_kwargs,
    dimension_combinations,
    get_format_input_bounds,
    get_reduce_extents,
    get_reduce_formats,
    get_supported_reduce_axioms,
    is_valid_reduce_dimension,
    use_int32_twos_complement,
)


@pytest.mark.perf
@parametrize(
    formats=get_reduce_formats,
    mathop=get_supported_reduce_axioms,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_bounds=get_format_input_bounds,
    reduce_pool=[ReducePool.Min, ReducePool.Max, ReducePool.Sum, ReducePool.Average],
    dimension_combinations=lambda mathop, dest_acc, formats: [
        dim
        for dim in dimension_combinations
        if is_valid_reduce_dimension(mathop, dest_acc, formats, dim)
    ],
    reduced_extent=get_reduce_extents,
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
    input_bounds,
    dimension_combinations,
    reduced_extent,
    run_types,
    loop_factor,
    is_perf,
):
    input_dimensions = dimension_combinations
    tile_count = input_dimensions[0] * input_dimensions[1] // (32 * 32)

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
            twos_complement=use_int32_twos_complement(formats, reduce_pool, mathop),
        ),
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
    )
