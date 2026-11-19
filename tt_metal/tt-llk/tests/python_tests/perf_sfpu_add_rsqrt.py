# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.llk_params import ApproximationMode, DestAccumulation
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_add_rsqrt import (
    ADDENDS,
    FORMATS,
    INPUT_SCALES,
)
from test_sfpu_add_rsqrt import test_sfpu_add_rsqrt as _test_sfpu_add_rsqrt
from test_sfpu_add_rsqrt import (
    test_sfpu_add_rsqrt_input_scale as _test_sfpu_add_rsqrt_input_scale,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    addend=list(ADDENDS),
    approx=[ApproximationMode.No, ApproximationMode.Yes],
    typed_bf16_store=[False, True],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_add_rsqrt(
    perf_report,
    formats,
    dest_acc,
    addend,
    approx,
    typed_bf16_store,
    run_types,
    loop_factor,
    is_perf,
):
    _test_sfpu_add_rsqrt(
        formats,
        dest_acc,
        addend,
        approx,
        typed_bf16_store,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_scale=list(INPUT_SCALES),
    approx=[ApproximationMode.No, ApproximationMode.Yes],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_add_rsqrt_input_scale(
    perf_report,
    formats,
    dest_acc,
    input_scale,
    approx,
    run_types,
    loop_factor,
    is_perf,
):
    _test_sfpu_add_rsqrt_input_scale(
        formats,
        dest_acc,
        input_scale,
        approx,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
