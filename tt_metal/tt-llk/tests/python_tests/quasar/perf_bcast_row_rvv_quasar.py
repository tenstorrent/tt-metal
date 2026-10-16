# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# One perf file per bcast source: a perf CSV holds one column schema and is keyed by its
# recorded sweep columns, which do not distinguish the sources (see BCAST_ROW_SOURCES).

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR
from quasar.test_bcast_row_scalar_quasar import (
    BCAST_ROW_SOURCES,
)
from quasar.test_bcast_row_scalar_quasar import (
    test_bcast_row_scalar_quasar as run_bcast_row_scalar,
)

SOURCE = "rvv"


@pytest.mark.perf
@pytest.mark.quasar
def test_perf_bcast_row_rvv_quasar(perf_report):
    run_bcast_row_scalar(
        SOURCE,
        run_types=BCAST_ROW_SOURCES[SOURCE]["perf_run_types"],
        loop_factor=PERF_LOOP_FACTOR_QUASAR,
        is_perf=True,
        perf_report=perf_report,
    )
