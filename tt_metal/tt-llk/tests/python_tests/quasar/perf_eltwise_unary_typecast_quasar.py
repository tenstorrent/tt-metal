# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    ApproximationMode,
    DestSync,
    MathOperation,
)
from helpers.param_config import parametrize
from quasar.test_eltwise_unary_sfpu_quasar import (
    PERF_INPUT_DIMENSIONS_UNARY,
    _dest_acc_for_typecast_formats,
    run_eltwise_unary_sfpu_quasar,
    typecast_formats,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    mathop=[MathOperation.Typecast],
    formats=typecast_formats(),
    dest_acc=_dest_acc_for_typecast_formats,
    approx_mode=[ApproximationMode.No],
    dest_sync=[DestSync.Half],
    input_dimensions=PERF_INPUT_DIMENSIONS_UNARY,
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_unary_typecast_quasar(
    perf_report,
    mathop,
    formats,
    dest_acc,
    approx_mode,
    dest_sync,
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
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
