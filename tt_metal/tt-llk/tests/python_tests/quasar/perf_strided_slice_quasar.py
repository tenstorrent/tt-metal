# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# One perf file per strided-slice source: a perf CSV holds one column schema and is keyed by
# its recorded sweep columns, which do not distinguish the sources (see STRIDED_SLICE_SOURCES).

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR
from quasar.test_strided_slice_quasar import (
    PERF_STRIDED_SLICE_PARAMS,
    STRIDED_SLICE_SOURCES,
)
from quasar.test_strided_slice_quasar import (
    test_strided_slice_quasar as run_strided_slice,
)

SOURCE = "llk"


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize("slice_start, slice_step", PERF_STRIDED_SLICE_PARAMS)
def test_perf_strided_slice_quasar(perf_report, slice_start, slice_step):
    run_strided_slice(
        SOURCE,
        slice_start,
        slice_step,
        run_types=STRIDED_SLICE_SOURCES[SOURCE]["perf_run_types"],
        loop_factor=PERF_LOOP_FACTOR_QUASAR,
        is_perf=True,
        perf_report=perf_report,
    )
