# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_pack_l1_acc import PACK_L1_ACC_SWEEP, _run_pack_l1_acc_test


@pytest.mark.perf
@parametrize(
    **PACK_L1_ACC_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
)
def test_perf_pack_l1_acc(
    perf_report,
    formats,
    dest_acc,
    l1_acc,
    tile_dimensions,
    tilize,
    dest_index,
    input_dimensions,
    run_types,
    loop_factor,
):
    _run_pack_l1_acc_test(
        formats,
        dest_acc,
        l1_acc,
        tile_dimensions,
        tilize,
        dest_index,
        input_dimensions,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
