# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR_4_TRISC,
)
from helpers.param_config import flat_axes, parametrize
from quasar.test_sfpu_exp_parallel_matmul_quasar import (
    ALL_SFPU_IMPLS,
    PARALLEL_MATMUL_EXP_FLAT_NAMES,
    SFPU_UNARY_FORMATS,
    generate_parallel_matmul_exp_combinations,
)
from quasar.test_sfpu_exp_parallel_matmul_quasar import (
    test_sfpu_exp_parallel_matmul_quasar as run_sfpu_exp_parallel_matmul,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **flat_axes(
        PARALLEL_MATMUL_EXP_FLAT_NAMES,
        generate_parallel_matmul_exp_combinations(
            SFPU_UNARY_FORMATS, is_perf=True, impls=ALL_SFPU_IMPLS
        ),
    ),
    run_types=PERF_RUN_TYPES_QUASAR_4_TRISC,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_sfpu_exp_parallel_matmul_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    sfpu_impl,
    approx_mode,
    exp_input_dimensions,
    input_A_dimensions,
    input_B_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_sfpu_exp_parallel_matmul(
        formats,
        dest_acc,
        dest_sync,
        sfpu_impl,
        approx_mode,
        exp_input_dimensions,
        input_A_dimensions,
        input_B_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
