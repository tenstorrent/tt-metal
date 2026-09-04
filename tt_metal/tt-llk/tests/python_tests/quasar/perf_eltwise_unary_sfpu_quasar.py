# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize
from quasar.test_eltwise_unary_sfpu_quasar import (
    generate_sfpu_unary_combinations,
)
from quasar.test_eltwise_unary_sfpu_quasar import (
    test_eltwise_unary_sfpu_quasar as run_eltwise_unary_sfpu_quasar,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    mathop_formats_dest_acc_sync_implied_math_input_dims=generate_sfpu_unary_combinations(
        is_perf=True
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_unary_sfpu_quasar(
    perf_report,
    mathop_formats_dest_acc_sync_implied_math_input_dims,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_unary_sfpu_quasar(
        [mathop_formats_dest_acc_sync_implied_math_input_dims],
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
