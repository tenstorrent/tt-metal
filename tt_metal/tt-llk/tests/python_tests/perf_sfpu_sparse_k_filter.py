# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_sparse_k_filter import (
    FIELD_LAYOUTS,
    ITERATIONS,
    OUT_SHIFTS,
    _run,
    _valid_my_banks,
)


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    layout=FIELD_LAYOUTS,
    my_bank=lambda layout: _valid_my_banks(layout),
    out_shift=OUT_SHIFTS,
    iterations=ITERATIONS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_sparse_k_filter(
    perf_report, layout, my_bank, out_shift, iterations, run_types, loop_factor, is_perf
):
    _run(
        layout,
        my_bank,
        out_shift,
        iterations,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
