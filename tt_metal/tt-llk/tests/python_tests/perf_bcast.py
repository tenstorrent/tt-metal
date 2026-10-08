# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_bcast import (
    BCAST_FORMATS,
    _run_unpack_bcast_test,
    get_perf_input_dimensions_bcast,
    get_valid_broadcast_types,
    get_valid_dest_acc_bcast,
    get_valid_perf_tile_dimensions_bcast,
)

PERF_LOOP_FACTOR = 32


@pytest.mark.perf
@parametrize(
    formats=BCAST_FORMATS,
    dest_acc=get_valid_dest_acc_bcast,
    tile_dimensions=get_valid_perf_tile_dimensions_bcast,
    broadcast_type=get_valid_broadcast_types,
    input_dimensions=get_perf_input_dimensions_bcast,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_unpack_bcast(
    perf_report,
    formats,
    dest_acc,
    tile_dimensions,
    broadcast_type,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _run_unpack_bcast_test(
        formats,
        dest_acc,
        tile_dimensions,
        broadcast_type,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
