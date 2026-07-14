# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize
from quasar.test_transpose_dest_quasar import (
    PERF_TRANSPOSE_DEST_COMBINATIONS,
)
from quasar.test_transpose_dest_quasar import (
    test_transpose_dest_quasar as run_transpose_dest,
)
from quasar.test_transpose_dest_quasar import (
    transpose_dest_implied_math_formats,
)

@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_transpose_dims=PERF_TRANSPOSE_DEST_COMBINATIONS,
    implied_math_format=lambda formats_dest_acc_sync_transpose_dims: transpose_dest_implied_math_formats(
        is_perf=True
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_transpose_dest_quasar(
    perf_report,
    formats_dest_acc_sync_transpose_dims,
    implied_math_format,
    run_types,
    loop_factor,
    is_perf,
):
    run_transpose_dest(
        formats_dest_acc_sync_transpose_dims,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
