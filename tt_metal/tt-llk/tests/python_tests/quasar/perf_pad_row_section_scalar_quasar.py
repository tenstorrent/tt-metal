# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# One perf file per pad-row source: a perf CSV holds one column schema and is keyed by its
# recorded sweep columns, which do not distinguish the sources (see PAD_ROW_SOURCES).

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR
from quasar.test_pad_row_section_quasar import (
    PAD_ROW_SOURCES,
)
from quasar.test_pad_row_section_quasar import (
    test_pad_row_section_quasar as run_pad_row_section,
)

SOURCE = "scalar"


@pytest.mark.perf
@pytest.mark.quasar
def test_perf_pad_row_section_scalar_quasar(perf_report):
    run_pad_row_section(
        SOURCE,
        run_types=PAD_ROW_SOURCES[SOURCE]["perf_run_types"],
        loop_factor=PERF_LOOP_FACTOR_QUASAR,
        is_perf=True,
        perf_report=perf_report,
    )
