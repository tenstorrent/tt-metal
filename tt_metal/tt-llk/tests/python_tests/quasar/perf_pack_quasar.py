# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.dest_params import dest_sync_modes
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize
from quasar.test_pack_quasar import (
    PACK_FORMATS,
    pack_dest_acc,
    pack_input_dimensions,
    pack_relu_types,
    pack_tile_dimensions,
    pack_unpack_to_dest,
)
from quasar.test_pack_quasar import test_pack_quasar as run_pack


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=PACK_FORMATS,
    dest_acc=pack_dest_acc,
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=pack_unpack_to_dest,
    relu_type=pack_relu_types,
    tile_dimensions=lambda formats, dest_acc: pack_tile_dimensions(
        formats, dest_acc, is_perf=True
    ),
    input_dimensions=lambda dest_acc, dest_sync, tile_dimensions: pack_input_dimensions(
        dest_acc, dest_sync, tile_dimensions, is_perf=True
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_pack_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    relu_type,
    tile_dimensions,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_pack(
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        relu_type,
        tile_dimensions,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
