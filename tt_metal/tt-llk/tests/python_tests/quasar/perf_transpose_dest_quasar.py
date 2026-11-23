# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import flat_axes, parametrize
from quasar.test_transpose_dest_quasar import (
    PERF_TRANSPOSE_DEST_COMBINATIONS,
    TRANSPOSE_DEST_FLAT_NAMES,
)
from quasar.test_transpose_dest_quasar import (
    test_transpose_dest_quasar as run_transpose_dest,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **flat_axes(TRANSPOSE_DEST_FLAT_NAMES, PERF_TRANSPOSE_DEST_COMBINATIONS),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_transpose_dest_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    math_transpose_faces,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_transpose_dest(
        formats,
        dest_acc,
        dest_sync,
        math_transpose_faces,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
