# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    ImpliedMathFormat,
)
from helpers.param_config import parametrize
from quasar.test_sfpu_where_quasar import (
    PERF_SFPU_WHERE_TEST_CASES,
    PERF_SFPU_WHERE_VECTOR_MODES,
    get_valid_formats_dest_acc,
)
from quasar.test_sfpu_where_quasar import (
    test_sfpu_where_quasar as run_sfpu_where_quasar,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc=get_valid_formats_dest_acc(),
    implied_math_format=[ImpliedMathFormat.Yes],
    test_case=PERF_SFPU_WHERE_TEST_CASES,
    vector_mode=PERF_SFPU_WHERE_VECTOR_MODES,
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_sfpu_where_quasar(
    perf_report,
    formats_dest_acc,
    implied_math_format,
    test_case,
    vector_mode,
    run_types,
    loop_factor,
    is_perf,
):
    run_sfpu_where_quasar(
        formats_dest_acc,
        implied_math_format,
        test_case,
        vector_mode,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
