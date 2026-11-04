# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PerfRunType
from helpers.param_config import parametrize
from quasar.test_isolate_sfpu_where_quasar import (
    generate_isolate_where_combinations,
)
from quasar.test_isolate_sfpu_where_quasar import (
    test_isolate_sfpu_where_quasar as run_isolate_sfpu_where_quasar,
)
from quasar.test_sfpu_where_quasar import PERF_SFPU_WHERE_TEST_CASES

# Only TRISC3 does work in this kernel; L1_TO_L1 already covers it.
PERF_RUN_TYPES_ISOLATE_SFPU = [[PerfRunType.L1_TO_L1]]


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc_input_dims=generate_isolate_where_combinations(is_perf=True),
    test_case=PERF_SFPU_WHERE_TEST_CASES,
    run_types=PERF_RUN_TYPES_ISOLATE_SFPU,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_isolate_sfpu_where_quasar(
    perf_report,
    formats_dest_acc_input_dims,
    test_case,
    run_types,
    loop_factor,
    is_perf,
):
    run_isolate_sfpu_where_quasar(
        formats_dest_acc_input_dims,
        test_case,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
