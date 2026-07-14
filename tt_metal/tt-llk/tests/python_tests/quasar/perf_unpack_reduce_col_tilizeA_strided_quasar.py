# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    PERF_UNPACK_REDUCE_COL_TILIZEA_STRIDED_COMBINATIONS,
)
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    test_unpack_reduce_col_tilizeA_strided_quasar as run_unpack_reduce_col_tilizeA_strided,
)
from quasar.test_unpack_reduce_col_tilizeA_strided_quasar import (
    unpack_reduce_col_tilizeA_strided_implied_math_formats,
)

@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel_dims=PERF_UNPACK_REDUCE_COL_TILIZEA_STRIDED_COMBINATIONS,
    implied_math_format=lambda: unpack_reduce_col_tilizeA_strided_implied_math_formats(
        is_perf=True
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_unpack_reduce_col_tilizeA_strided_quasar(
    perf_report,
    formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel_dims,
    implied_math_format,
    run_types,
    loop_factor,
    is_perf,
):
    run_unpack_reduce_col_tilizeA_strided(
        formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel_dims,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
