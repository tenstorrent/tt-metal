# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_valid_math_fidelities
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
)
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_binary_reuse_dest import (
    DEST_REUSE_MATH_OPS,
    _run_eltwise_binary_dest_reuse_test,
    get_dest_reuse_formats,
    get_dest_reuse_perf_input_dimensions,
    get_dest_reuse_perf_output_dimensions,
    get_dest_reuse_perf_tile_dimensions,
)

PERF_LOOP_FACTOR = 32


@pytest.mark.perf
@parametrize(
    reuse_dest_type=[
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ],
    mathop=DEST_REUSE_MATH_OPS,
    formats=get_dest_reuse_formats,
    dest_acc=[DestAccumulation.No],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(formats, mathop),
    tile_dimensions=lambda: get_dest_reuse_perf_tile_dimensions(),
    input_dimensions=get_dest_reuse_perf_input_dimensions,
    output_dimensions=get_dest_reuse_perf_output_dimensions,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_eltwise_binary_reuse_dest(
    perf_report,
    reuse_dest_type,
    mathop,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    math_fidelity,
    tile_dimensions,
    input_dimensions,
    output_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _run_eltwise_binary_dest_reuse_test(
        reuse_dest_type,
        mathop,
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        math_fidelity,
        tile_dimensions,
        input_dimensions,
        output_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
