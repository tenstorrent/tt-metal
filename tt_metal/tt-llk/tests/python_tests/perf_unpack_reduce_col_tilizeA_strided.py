# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import DestSync
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from helpers.tile_constants import MAX_FACE_R_DIM
from test_unpack_reduce_col_tilizeA_strided import (
    UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS,
    UNPACK_REDUCE_COL_TILIZEA_STRIDED_NUM_FACES,
    UNPACK_REDUCE_COL_TILIZEA_STRIDED_POOL_TYPES,
)
from test_unpack_reduce_col_tilizeA_strided import (
    test_unpack_reduce_col_tilizeA_strided as run_unpack_reduce_col_tilizeA_strided,
)
from test_unpack_reduce_col_tilizeA_strided import (
    unpack_reduce_col_tilizeA_strided_dest_acc,
    unpack_reduce_col_tilizeA_strided_max_tiles_per_iter,
)


@pytest.mark.perf
@parametrize(
    formats=UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS,
    dest_acc=unpack_reduce_col_tilizeA_strided_dest_acc,
    max_tiles_per_iter=unpack_reduce_col_tilizeA_strided_max_tiles_per_iter,
    num_faces=UNPACK_REDUCE_COL_TILIZEA_STRIDED_NUM_FACES,
    pool_type=UNPACK_REDUCE_COL_TILIZEA_STRIDED_POOL_TYPES,
    dest_sync=[DestSync.Half],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
)
def test_perf_unpack_reduce_col_tilizeA_strided(
    perf_report,
    formats,
    dest_acc,
    max_tiles_per_iter,
    num_faces,
    pool_type,
    dest_sync,
    run_types,
    loop_factor,
):
    run_unpack_reduce_col_tilizeA_strided(
        formats,
        dest_acc,
        max_tiles_per_iter,
        num_faces,
        pool_type,
        MAX_FACE_R_DIM,
        dest_sync,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
