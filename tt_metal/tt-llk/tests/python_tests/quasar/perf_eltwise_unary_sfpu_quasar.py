# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize, runtime
from quasar.test_eltwise_unary_sfpu_quasar import (
    sfpu_unary_approx_modes,
    sfpu_unary_dest_acc_modes,
    sfpu_unary_dest_sync_modes,
    sfpu_unary_formats,
    sfpu_unary_implied_math_formats,
    sfpu_unary_input_dimensions,
    sfpu_unary_mathops,
)
from quasar.test_eltwise_unary_sfpu_quasar import (
    test_eltwise_unary_sfpu_quasar as run_eltwise_unary_sfpu_quasar,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    mathop=sfpu_unary_mathops(),
    formats=sfpu_unary_formats,
    dest_acc=sfpu_unary_dest_acc_modes,
    dest_sync_mode=lambda mathop: sfpu_unary_dest_sync_modes(mathop, is_perf=True),
    implied_math_format=lambda: sfpu_unary_implied_math_formats(is_perf=True),
    approx_mode=sfpu_unary_approx_modes,
    input_dimensions=runtime(
        lambda mathop, dest_acc: sfpu_unary_input_dimensions(
            mathop, dest_acc, is_perf=True
        )
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_unary_sfpu_quasar(
    perf_report,
    mathop,
    formats,
    dest_acc,
    dest_sync_mode,
    implied_math_format,
    approx_mode,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_unary_sfpu_quasar(
        mathop,
        formats,
        dest_acc,
        dest_sync_mode,
        implied_math_format,
        approx_mode,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
