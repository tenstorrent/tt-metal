# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize, runtime
from quasar.test_unpack_tilize_quasar import (
    UNPACK_TILIZE_FORMATS,
)
from quasar.test_unpack_tilize_quasar import (
    test_unpack_tilize_quasar as run_unpack_tilize,
)
from quasar.test_unpack_tilize_quasar import (
    unpack_tilize_dest_acc_modes,
    unpack_tilize_dest_sync_modes,
    unpack_tilize_formats,
    unpack_tilize_input_dimensions,
    unpack_tilize_tile_dimensions,
    unpack_tilize_unpacker_engines,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=lambda: unpack_tilize_formats(UNPACK_TILIZE_FORMATS, is_perf=True),
    dest_acc=unpack_tilize_dest_acc_modes,
    dest_sync_mode=lambda: unpack_tilize_dest_sync_modes(is_perf=True),
    unpacker_sel=unpack_tilize_unpacker_engines,
    tile_dimensions=runtime(
        lambda formats, dest_acc: unpack_tilize_tile_dimensions(
            formats, dest_acc, is_perf=True
        )
    ),
    input_dimensions=runtime(
        lambda dest_acc, dest_sync_mode, tile_dimensions: unpack_tilize_input_dimensions(
            dest_acc, dest_sync_mode, tile_dimensions, is_perf=True
        )
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_unpack_tilize_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync_mode,
    unpacker_sel,
    input_dimensions,
    tile_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_unpack_tilize(
        formats,
        dest_acc,
        dest_sync_mode,
        unpacker_sel,
        input_dimensions,
        tile_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
