# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_eltwise_unary_datacopy_custom import (
    test_unary_datacopy_custom as _test_unary_datacopy_custom,
)


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_unary_datacopy_custom(
    perf_report, formats, dest_acc, run_types, loop_factor, is_perf
):
    _test_unary_datacopy_custom(
        formats,
        dest_acc,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
