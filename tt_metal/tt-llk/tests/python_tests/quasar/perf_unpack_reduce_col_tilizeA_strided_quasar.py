# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import flat_axes, parametrize
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    PERF_UNPACK_REDUCE_COL_TILIZEA_STRIDED_COMBINATIONS,
    UNPACK_REDUCE_COL_TILIZEA_STRIDED_FLAT_NAMES,
)
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    test_unpack_reduce_col_tilizeA_strided_quasar as run_unpack_reduce_col_tilizeA_strided,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **flat_axes(
        UNPACK_REDUCE_COL_TILIZEA_STRIDED_FLAT_NAMES,
        PERF_UNPACK_REDUCE_COL_TILIZEA_STRIDED_COMBINATIONS,
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_unpack_reduce_col_tilizeA_strided_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    input_dimensions,
    pool_type,
    tile_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_unpack_reduce_col_tilizeA_strided(
        formats,
        dest_acc,
        dest_sync,
        input_dimensions,
        pool_type,
        tile_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
