# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR
from helpers.param_config import flat_axes, parametrize
from quasar.test_dest_sync_mechanism_quasar import (
    DEST_SYNC_MECHANISM_FLAT_NAMES,
    DEST_SYNC_MECHANISM_PERF_RUN_TYPES,
    PERF_DEST_SYNC_MECHANISM_COMBINATIONS,
)
from quasar.test_dest_sync_mechanism_quasar import (
    test_dest_sync_mechanism_quasar as run_dest_sync_mechanism,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **flat_axes(DEST_SYNC_MECHANISM_FLAT_NAMES, PERF_DEST_SYNC_MECHANISM_COMBINATIONS),
    run_types=DEST_SYNC_MECHANISM_PERF_RUN_TYPES,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_dest_sync_mechanism_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    dest_sync_mechanism,
    tiles_per_section,
    run_types,
    loop_factor,
    is_perf,
):
    run_dest_sync_mechanism(
        formats,
        dest_acc,
        dest_sync,
        dest_sync_mechanism,
        tiles_per_section,
        run_types,
        loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
