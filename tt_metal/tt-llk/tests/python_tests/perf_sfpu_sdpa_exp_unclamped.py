# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_sdpa_exp_unclamped import (
    BF16_HALF,
    BF16_ONE,
    FORMATS,
    INPUT_RANGES,
    test_sfpu_sdpa_exp_unclamped,
)


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    # input_range is stimulus-only; scale_en and value_bits stay recorded.
    input_range=[INPUT_RANGES[0]],
    scale_en=[False, True],
    scale=[BF16_ONE, BF16_HALF],
    num_tiles=[1, 2],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_sfpu_sdpa_exp_unclamped(
    perf_report,
    formats,
    input_range,
    scale_en,
    scale,
    num_tiles,
    run_types,
    loop_factor,
    is_perf,
):
    test_sfpu_sdpa_exp_unclamped(
        formats,
        input_range,
        scale_en,
        scale,
        num_tiles,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
