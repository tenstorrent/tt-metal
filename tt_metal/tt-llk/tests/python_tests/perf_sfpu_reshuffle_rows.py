# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""reshuffle_rows perf, sharing sources/sfpu_reshuffle_rows_test.cpp with the functional test.

The kernel cost depends on the mask, so perf runs a subset of the functional masks: every row
valid, three of four rows per 4-row group valid, and every row skipped.
"""

import pytest
from conftest import skip_for_quasar
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_sfpu_reshuffle_rows import RESHUFFLE_ROWS_SWEEP, _run_sfpu_reshuffle_rows

PERF_MASKS = ["identity", "skip_last_row_of_group", "all_skip"]


@pytest.mark.perf
@skip_for_quasar
@parametrize(
    **{**RESHUFFLE_ROWS_SWEEP, "mask_name": PERF_MASKS},
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
)
def test_perf_sfpu_reshuffle_rows(
    perf_report, formats, dest_acc, mask_name, run_types, loop_factor
):
    _run_sfpu_reshuffle_rows(
        formats,
        dest_acc,
        mask_name,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
