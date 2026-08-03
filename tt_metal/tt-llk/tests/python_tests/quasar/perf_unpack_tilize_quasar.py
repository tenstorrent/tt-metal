# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize
from quasar.test_unpack_tilize_quasar import (
    PERF_UNPACK_TILIZE_COMBINATIONS,
)
from quasar.test_unpack_tilize_quasar import (
    test_unpack_tilize_quasar as run_unpack_tilize,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_unpack_sel_dimensions=PERF_UNPACK_TILIZE_COMBINATIONS,
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_unpack_tilize_quasar(
    perf_report,
    formats_dest_acc_sync_unpack_sel_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_unpack_tilize(
        formats_dest_acc_sync_unpack_sel_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
