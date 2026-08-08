# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    DataCopyType,
)
from helpers.param_config import parametrize, runtime
from quasar.test_eltwise_unary_datacopy_quasar import (
    DATACOPY_FORMATS,
    datacopy_dest_acc_modes,
    datacopy_dest_indices,
    datacopy_dest_sync_modes,
    datacopy_implied_math_formats,
    datacopy_input_dimensions,
    datacopy_tile_dimensions,
)
from quasar.test_eltwise_unary_datacopy_quasar import (
    test_eltwise_unary_datacopy_quasar as run_eltwise_unary_datacopy,
)


@pytest.mark.nightly
@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=DATACOPY_FORMATS,
    dest_acc=datacopy_dest_acc_modes,
    data_copy_type=[DataCopyType.A2D, DataCopyType.B2D],
    dest_sync_mode=lambda: datacopy_dest_sync_modes(is_perf=True),
    tile_dimensions=runtime(
        lambda formats: datacopy_tile_dimensions(formats, is_perf=True)
    ),
    input_dimensions=runtime(
        lambda dest_acc, dest_sync_mode, tile_dimensions: datacopy_input_dimensions(
            dest_acc, dest_sync_mode, tile_dimensions, is_perf=True
        )
    ),
    dest_index=runtime(
        lambda dest_acc, dest_sync_mode, input_dimensions, tile_dimensions: datacopy_dest_indices(
            dest_acc,
            dest_sync_mode,
            input_dimensions,
            tile_dimensions,
            is_perf=True,
        )
    ),
    implied_math_format=lambda formats: datacopy_implied_math_formats(
        formats, is_perf=True
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_unary_datacopy_quasar(
    perf_report,
    formats,
    dest_acc,
    data_copy_type,
    input_dimensions,
    dest_sync_mode,
    dest_index,
    tile_dimensions,
    implied_math_format,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_unary_datacopy(
        formats,
        dest_acc,
        data_copy_type,
        input_dimensions,
        dest_sync_mode,
        dest_index,
        tile_dimensions,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
