# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ALL_MATH_FIDELITIES,
    BroadcastType,
    DestAccumulation,
    MathOperation,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_bcast_col_custom import _run_eltwise_bcast_col_custom

_TILE_ROWS = [32, 16]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
        ]
    ),
    mathop=[MathOperation.Elwsub, MathOperation.Elwmul],
    dest_acc=[DestAccumulation.No],
    math_fidelity=ALL_MATH_FIDELITIES,
    broadcast_type=[BroadcastType.Column],
    tile_rows=_TILE_ROWS,
    input_width=list(range(32, 257, 32)) + [1024],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_eltwise_bcast_col_custom(
    perf_report,
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    broadcast_type,
    tile_rows,
    input_width,
    run_types,
    loop_factor,
    is_perf,
):
    _run_eltwise_bcast_col_custom(
        formats,
        mathop,
        dest_acc,
        math_fidelity,
        broadcast_type,
        tile_rows,
        input_width,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
