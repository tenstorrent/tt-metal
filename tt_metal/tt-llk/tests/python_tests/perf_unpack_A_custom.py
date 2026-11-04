# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_unpack_A_custom import UNPACK_A_CUSTOM_FORMATS
from test_unpack_A_custom import test_unpack_A_custom as _test_unpack_A_custom


@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=UNPACK_A_CUSTOM_FORMATS,
    input_dimensions=[[32, 32], [64, 64], [32, 256]],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_unpack_A_custom(
    perf_report, formats, input_dimensions, run_types, loop_factor, is_perf
):
    _test_unpack_A_custom(
        formats,
        input_dimensions,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
