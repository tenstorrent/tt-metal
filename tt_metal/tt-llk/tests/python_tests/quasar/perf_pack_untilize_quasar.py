# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize, runtime
from quasar.test_pack_untilize_quasar import (
    PACK_UNTILIZE_FORMATS,
    pack_untilize_dest_acc_modes,
    pack_untilize_dest_sync_modes,
    pack_untilize_formats,
    pack_untilize_input_dimensions,
    pack_untilize_tile_dimensions,
)
from quasar.test_pack_untilize_quasar import (
    test_pack_untilize_quasar as run_pack_untilize,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=lambda: pack_untilize_formats(PACK_UNTILIZE_FORMATS),
    dest_acc=pack_untilize_dest_acc_modes,
    dest_sync_mode=lambda: pack_untilize_dest_sync_modes(is_perf=True),
    tile_dimensions=runtime(pack_untilize_tile_dimensions),
    input_dimensions=runtime(
        lambda dest_acc, dest_sync_mode, tile_dimensions: pack_untilize_input_dimensions(
            dest_acc, dest_sync_mode, tile_dimensions, is_perf=True
        )
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_pack_untilize_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync_mode,
    input_dimensions,
    tile_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_pack_untilize(
        formats,
        dest_acc,
        dest_sync_mode,
        input_dimensions,
        tile_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
