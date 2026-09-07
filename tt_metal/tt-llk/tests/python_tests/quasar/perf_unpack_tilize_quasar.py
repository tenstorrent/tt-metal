# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.dest_params import dest_sync_modes
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize
from quasar.test_unpack_tilize_quasar import (
    UNPACK_TILIZE_FORMATS,
)
from quasar.test_unpack_tilize_quasar import (
    test_unpack_tilize_quasar as run_unpack_tilize,
)
from quasar.test_unpack_tilize_quasar import (
    unpack_tilize_dest_acc,
    unpack_tilize_input_dimensions,
    unpack_tilize_tile_dimensions,
    unpack_tilize_unpack_to_dest,
    unpack_tilize_unpacker_sel,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=UNPACK_TILIZE_FORMATS,
    dest_acc=unpack_tilize_dest_acc,
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=unpack_tilize_unpack_to_dest,
    unpacker_sel=unpack_tilize_unpacker_sel,
    tile_dimensions=lambda formats, dest_acc: unpack_tilize_tile_dimensions(
        formats, dest_acc, is_perf=True
    ),
    input_dimensions=lambda dest_acc, dest_sync, tile_dimensions: unpack_tilize_input_dimensions(
        dest_acc, dest_sync, tile_dimensions, is_perf=True
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_unpack_tilize_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    unpacker_sel,
    tile_dimensions,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_unpack_tilize(
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        unpacker_sel,
        tile_dimensions,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
