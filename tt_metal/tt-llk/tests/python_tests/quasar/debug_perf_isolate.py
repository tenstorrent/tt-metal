# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused repro: run each PerfRunType in isolation so a deadlock surfaces
per-scenario instead of inside the 5-in-1 sweep.

Uses the real failing shape ([32,32] x [32,128] => CT_DIM=4, RT_DIM=1, KT_DIM=1,
loop_factor=32). The asymmetric CT!=RT shape is what exercises the mock's
reuse/reload split; the symmetric 1x1x1 shape does not and passes cleanly."""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    PerfRunType,
    Transpose,
)
from quasar.test_matmul_quasar import test_matmul as run_matmul


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize(
    "run_type",
    [
        pytest.param(PerfRunType.L1_TO_L1, id="L1_TO_L1"),
        pytest.param(PerfRunType.UNPACK_ISOLATE, id="UNPACK_ISOLATE"),
        pytest.param(PerfRunType.MATH_ISOLATE, id="MATH_ISOLATE"),
        pytest.param(PerfRunType.PACK_ISOLATE, id="PACK_ISOLATE"),
        pytest.param(PerfRunType.L1_CONGESTION, id="L1_CONGESTION"),
    ],
)
def test_debug_perf_isolate(run_type, perf_report):
    run_matmul(
        MathFidelity.LoFi,
        DestSync.Half,
        DestAccumulation.No,
        ([TILE_DIM, TILE_DIM], [TILE_DIM, 4 * TILE_DIM]),
        InputOutputFormat(DataFormat.Float16, DataFormat.Float16),
        ImpliedMathFormat.Yes,
        None,
        False,
        Transpose.No,
        [run_type],
        32,
        is_perf=True,
        perf_report=perf_report,
    )
