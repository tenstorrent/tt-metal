# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_zero_pad import FORMATS, ROW_RANGES, test_sfpu_zero_pad


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
    dest_acc=get_valid_dest_accumulation_modes,
    row_range=ROW_RANGES,
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_sfpu_zero_pad(
    perf_report, formats, dest_acc, row_range, run_types, loop_factor, is_perf
):
    test_sfpu_zero_pad(
        formats,
        dest_acc,
        row_range,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
