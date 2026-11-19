# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.param_config import parametrize, runtime
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_sdpa_exp_unclamped import (
    BF16_HALF,
    BF16_ONE,
    FORMATS,
    INPUT_RANGES,
)
from test_sfpu_sdpa_exp_unclamped import (
    test_sfpu_sdpa_exp_unclamped as _test_sfpu_sdpa_exp_unclamped,
)

PERF_INPUT_RANGE = INPUT_RANGES[0]


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    scale_en=[False, True],
    scale=lambda scale_en: [BF16_ONE] if not scale_en else [BF16_ONE, BF16_HALF],
    num_tiles=runtime([1, 2]),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_sdpa_exp_unclamped(
    perf_report,
    formats,
    scale_en,
    scale,
    num_tiles,
    run_types,
    loop_factor,
    is_perf,
):
    _test_sfpu_sdpa_exp_unclamped(
        formats,
        PERF_INPUT_RANGE,
        scale_en,
        scale,
        num_tiles,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
