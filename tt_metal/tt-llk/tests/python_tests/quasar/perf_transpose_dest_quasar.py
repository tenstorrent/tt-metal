# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    ImpliedMathFormat,
    Transpose,
)
from helpers.param_config import parametrize, runtime
from quasar.test_transpose_dest_quasar import (
    TRANSPOSE_DEST_FORMATS,
)
from quasar.test_transpose_dest_quasar import (
    test_transpose_dest_quasar as run_transpose_dest,
)
from quasar.test_transpose_dest_quasar import (
    transpose_dest_acc_modes,
    transpose_dest_formats,
    transpose_dest_input_dimensions,
    transpose_dest_sync_modes,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=lambda: transpose_dest_formats(TRANSPOSE_DEST_FORMATS),
    dest_acc=transpose_dest_acc_modes,
    dest_sync_mode=lambda: transpose_dest_sync_modes(is_perf=True),
    transpose=[Transpose.No, Transpose.Yes],
    input_dimensions=runtime(
        lambda dest_acc, dest_sync_mode: transpose_dest_input_dimensions(
            dest_acc, dest_sync_mode, is_perf=True
        )
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_transpose_dest_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync_mode,
    transpose,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_transpose_dest(
        formats,
        dest_acc,
        dest_sync_mode,
        transpose,
        input_dimensions,
        ImpliedMathFormat.Yes,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
