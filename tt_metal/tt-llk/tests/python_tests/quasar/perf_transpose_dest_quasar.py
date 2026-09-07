# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.dest_params import dest_sync_modes
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR, Transpose
from helpers.param_config import parametrize
from quasar.test_transpose_dest_quasar import (
    TRANSPOSE_DEST_FORMATS,
)
from quasar.test_transpose_dest_quasar import (
    test_transpose_dest_quasar as run_transpose_dest,
)
from quasar.test_transpose_dest_quasar import (
    transpose_dest_dest_acc,
    transpose_dest_implied_math_formats,
    transpose_dest_input_dimensions,
    transpose_dest_unpack_to_dest,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=TRANSPOSE_DEST_FORMATS,
    dest_acc=transpose_dest_dest_acc,
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=transpose_dest_unpack_to_dest,
    math_transpose_faces=[Transpose.No, Transpose.Yes],
    input_dimensions=lambda dest_acc, dest_sync: transpose_dest_input_dimensions(
        dest_acc, dest_sync, is_perf=True
    ),
    implied_math_format=lambda: transpose_dest_implied_math_formats(is_perf=True),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_transpose_dest_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    math_transpose_faces,
    input_dimensions,
    implied_math_format,
    run_types,
    loop_factor,
    is_perf,
):
    run_transpose_dest(
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        math_transpose_faces,
        input_dimensions,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
