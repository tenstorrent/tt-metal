# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR_4_TRISC,
)
from helpers.param_config import flat_axes, parametrize
from quasar.test_sfpu_add_parallel_matmul_quasar import (
    PARALLEL_MATMUL_ADD_FLAT_NAMES,
    SFPU_ADD_FORMATS,
    generate_parallel_matmul_add_combinations,
)
from quasar.test_sfpu_add_parallel_matmul_quasar import (
    test_sfpu_add_parallel_matmul_quasar as run_sfpu_add_parallel_matmul,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **flat_axes(
        PARALLEL_MATMUL_ADD_FLAT_NAMES,
        generate_parallel_matmul_add_combinations(SFPU_ADD_FORMATS, is_perf=True),
    ),
    run_types=PERF_RUN_TYPES_QUASAR_4_TRISC,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_sfpu_add_parallel_matmul_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    add_input_dimensions,
    input_A_dimensions,
    input_B_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_sfpu_add_parallel_matmul(
        formats,
        dest_acc,
        dest_sync,
        add_input_dimensions,
        input_A_dimensions,
        input_B_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
