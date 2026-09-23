# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import test_eltwise_binary_sfpu as _func
from helpers.llk_params import ApproximationMode
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES

_PERF_AXES = dict(
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[16],
    iterations=[32],
    approx_mode=[
        ApproximationMode.Yes,
        ApproximationMode.No,
    ],
    is_perf=[True],
)


def _perf_kwargs(perf_report, run_types, loop_factor, iterations, approx_mode, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        iterations=iterations,
        approx_mode=approx_mode,
    )


@pytest.mark.perf
@parametrize(
    **_func.FLOAT_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_float(
    perf_report,
    formats,
    dest_acc,
    mathop,
    bcast_dim,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float(
        formats,
        dest_acc,
        mathop,
        bcast_dim,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.DIV_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_div(
    perf_report,
    formats,
    dest_acc,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_div(
        formats,
        dest_acc,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.FLOAT_EXTENDED_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_float_extended(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float_extended(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.MASK_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_mask(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_mask(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.ATAN2_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_atan2(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_atan2(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.EQ_NE_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_eq_ne(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_eq_ne(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.FLOAT_COMPARISON_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_float_comparison(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float_comparison(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.ISCLOSE_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_isclose(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_isclose(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.LOGSIGMOID_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_logsigmoid(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_logsigmoid(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.INT_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_int(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_int(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.BITWISE_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_bitwise(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_bitwise(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.INT_UNIFORM_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_int_uniform(
    perf_report,
    mathop,
    dest_acc,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_int_uniform(
        mathop,
        dest_acc,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.RSUB_INT32_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_rsub_int32(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_rsub_int32(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.EQ_NE_INT_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_eq_ne_int(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_eq_ne_int(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.ADD_TOP_ROW_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_add_top_row(
    perf_report,
    formats,
    dest_acc,
    mathop,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_add_top_row(
        formats,
        dest_acc,
        mathop,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )


@pytest.mark.perf
@parametrize(
    **_func.BCAST_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[16],
    iterations=[32],
    # Kernel never reads APPROX_MODE; pin No so the schema column stays present
    # without compiling two identical ELFs.
    approx_mode=[ApproximationMode.No],
    is_perf=[True],
)
def test_perf_eltwise_binary_sfpu_bcast(
    perf_report,
    formats,
    bcast_dim,
    mathop,
    dest_acc,
    run_types,
    loop_factor,
    iterations,
    approx_mode,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_bcast(
        formats,
        bcast_dim,
        mathop,
        dest_acc,
        **_perf_kwargs(
            perf_report, run_types, loop_factor, iterations, approx_mode, is_perf
        ),
    )
