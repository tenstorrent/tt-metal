# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# SFPU bcast ROW with constant and runtime (compiler-opaque) dest tile indices.

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import BroadcastType as LlkBroadcastType
from helpers.llk_params import DestAccumulation, MathOperation
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_sfpu_binary_bcast_layout import run_bcast_layout

_PERF_CASES = [
    (fmt, dest_acc, LlkBroadcastType.Row, op, "L012")
    for fmt, dest_acc in (
        (DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float32, DestAccumulation.Yes),
        (DataFormat.Bfp8_b, DestAccumulation.No),
    )
    for op in (MathOperation.SfpuElwadd, MathOperation.SfpuElwmul)
]


@pytest.mark.perf
@parametrize(
    case=_PERF_CASES,
    bcast_l_runtime=[0, 1],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[16],
)
def test_perf_sfpu_binary_bcast_layout(
    perf_report, case, bcast_l_runtime, run_types, loop_factor
):
    run_bcast_layout(
        case,
        bcast_l_runtime=bcast_l_runtime,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
