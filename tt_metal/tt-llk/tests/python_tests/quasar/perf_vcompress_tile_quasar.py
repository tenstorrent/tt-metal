# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# One perf file per compress source and shape: a perf CSV holds one column schema and is
# keyed by its recorded sweep columns, which do not distinguish the row test from the
# tile test, so each gets its own file (see VCOMPRESS_SOURCES / VCOMPRESS_TILE_SOURCES).

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR
from quasar.test_vcompress_elems_quasar import (
    VCOMPRESS_TILE_SOURCES,
)
from quasar.test_vcompress_elems_quasar import (
    test_vcompress_tile_quasar as run_vcompress_tile,
)

SOURCE = "llk"


@pytest.mark.perf
@pytest.mark.quasar
def test_perf_vcompress_tile_quasar(perf_report):
    run_vcompress_tile(
        SOURCE,
        run_types=VCOMPRESS_TILE_SOURCES[SOURCE]["perf_run_types"],
        loop_factor=PERF_LOOP_FACTOR_QUASAR,
        is_perf=True,
        perf_report=perf_report,
    )
