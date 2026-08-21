# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize, runtime
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    PERF_UNPACK_REDUCE_COL_TILIZEA_STRIDED_COMBINATIONS,
)
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    test_unpack_reduce_col_tilizeA_strided_quasar as run_unpack_reduce_col_tilizeA_strided,
)
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    unpack_reduce_col_tilizeA_strided_implied_math_formats,
    unpack_reduce_col_tilizeA_strided_runtime_shapes,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel=PERF_UNPACK_REDUCE_COL_TILIZEA_STRIDED_COMBINATIONS,
    implied_math_format=unpack_reduce_col_tilizeA_strided_implied_math_formats,
    dimensions_and_tile=runtime(
        lambda formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel: unpack_reduce_col_tilizeA_strided_runtime_shapes(
            formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel, is_perf=True
        )
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_unpack_reduce_col_tilizeA_strided_quasar(
    perf_report,
    formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel,
    implied_math_format,
    dimensions_and_tile,
    run_types,
    loop_factor,
    is_perf,
):
    run_unpack_reduce_col_tilizeA_strided(
        formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel,
        implied_math_format,
        dimensions_and_tile,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
