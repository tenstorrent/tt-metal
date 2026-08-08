# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize, runtime
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS,
)
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    test_unpack_reduce_col_tilizeA_strided_quasar as run_unpack_reduce_col_tilizeA_strided,
)
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    unpack_reduce_col_tilizeA_strided_dest_acc_modes,
    unpack_reduce_col_tilizeA_strided_dest_sync_modes,
    unpack_reduce_col_tilizeA_strided_dimensions,
    unpack_reduce_col_tilizeA_strided_formats,
    unpack_reduce_col_tilizeA_strided_implied_math_formats,
    unpack_reduce_col_tilizeA_strided_pool_types,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=lambda: unpack_reduce_col_tilizeA_strided_formats(
        UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS
    ),
    dest_acc=unpack_reduce_col_tilizeA_strided_dest_acc_modes,
    dest_sync_mode=lambda: unpack_reduce_col_tilizeA_strided_dest_sync_modes(
        is_perf=True
    ),
    input_dimensions=runtime(
        lambda dest_acc, dest_sync_mode: unpack_reduce_col_tilizeA_strided_dimensions(
            dest_acc, dest_sync_mode, is_perf=True
        )
    ),
    pool_type=unpack_reduce_col_tilizeA_strided_pool_types,
    implied_math_format=unpack_reduce_col_tilizeA_strided_implied_math_formats,
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_unpack_reduce_col_tilizeA_strided_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync_mode,
    input_dimensions,
    pool_type,
    implied_math_format,
    run_types,
    loop_factor,
    is_perf,
):
    run_unpack_reduce_col_tilizeA_strided(
        formats,
        dest_acc,
        dest_sync_mode,
        input_dimensions,
        pool_type,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
