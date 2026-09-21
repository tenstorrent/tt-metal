# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    DestSync,
    ImpliedMathFormat,
)
from helpers.param_config import parametrize
from quasar.test_eltwise_unary_sfpu_quasar import (
    COMP_MATHOPS,
    MAIN_MATHOPS,
    PERF_INPUT_DIMENSIONS_UNARY,
    QSR_EXTRA_MATHOPS,
    _approx_modes_for_mathop,
    _dest_acc_for_mathop_formats,
    _float_formats_for_mathop,
    _int_comp_formats_for_mathop,
    run_eltwise_unary_sfpu_quasar,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    mathop=MAIN_MATHOPS,
    formats=_float_formats_for_mathop,
    dest_acc=_dest_acc_for_mathop_formats,
    approx_mode=_approx_modes_for_mathop,
    dest_sync=[DestSync.Half],
    implied_math_format=[ImpliedMathFormat.Yes],
    input_dimensions=PERF_INPUT_DIMENSIONS_UNARY,
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_unary_sfpu_quasar(
    perf_report,
    mathop,
    formats,
    dest_acc,
    approx_mode,
    dest_sync,
    implied_math_format,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_unary_sfpu_quasar(
        formats,
        dest_acc,
        mathop,
        approx_mode,
        input_dimensions,
        dest_sync,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    mathop=COMP_MATHOPS,
    formats=_int_comp_formats_for_mathop,
    dest_acc=_dest_acc_for_mathop_formats,
    approx_mode=_approx_modes_for_mathop,
    dest_sync=[DestSync.Half],
    implied_math_format=[ImpliedMathFormat.Yes],
    input_dimensions=PERF_INPUT_DIMENSIONS_UNARY,
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_unary_sfpu_comp_quasar(
    perf_report,
    mathop,
    formats,
    dest_acc,
    approx_mode,
    dest_sync,
    implied_math_format,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_unary_sfpu_quasar(
        formats,
        dest_acc,
        mathop,
        approx_mode,
        input_dimensions,
        dest_sync,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    mathop=QSR_EXTRA_MATHOPS,
    formats=_float_formats_for_mathop,
    dest_acc=_dest_acc_for_mathop_formats,
    approx_mode=_approx_modes_for_mathop,
    dest_sync=[DestSync.Half],
    implied_math_format=[ImpliedMathFormat.Yes],
    input_dimensions=PERF_INPUT_DIMENSIONS_UNARY,
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_unary_sfpu_qsr_extra_quasar(
    perf_report,
    mathop,
    formats,
    dest_acc,
    approx_mode,
    dest_sync,
    implied_math_format,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_unary_sfpu_quasar(
        formats,
        dest_acc,
        mathop,
        approx_mode,
        input_dimensions,
        dest_sync,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
