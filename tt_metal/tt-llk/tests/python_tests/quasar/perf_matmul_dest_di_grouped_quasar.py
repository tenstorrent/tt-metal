# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf for the grouped (multi-replay) dest-addressed DI path.

Instruction accounting says the grouped path's best case is `rt > ct >= 2`, where the baseline pays
two dest re-base instructions per output column: 14 overhead instructions per block against 6. The
shared perf sweep uses `exact_dest_fill`, which does not generate those shapes, so its grouped
numbers have never been taken.

Shapes and fidelities mirror test_matmul_dest_di_grouped_quasar.py so a perf row always has a
correctness row behind it. HiFi4 is the control: group_tiles=1 on a multi-tile block, so the op
falls back to the counter path and its rows must not move.

Read mean(MATH_ISOLATE) for the instruction saving and mean(L1_TO_L1) for whether it reaches the
kernel. Two loop factors separate a fixed per-kernel cost (decays as 1/N) from a per-tile one
(constant).
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    PERF_RUN_TYPES_QUASAR,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    Transpose,
)
from helpers.param_config import parametrize
from quasar.test_matmul_dest_di_grouped_quasar import GROUPED_DIMENSIONS
from quasar.test_matmul_quasar import test_matmul as run_matmul


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    format=[InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)],
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4],
    dest_sync_mode=[DestSync.Half],
    dest_acc=[DestAccumulation.No],
    dimensions=GROUPED_DIMENSIONS,
    implied_math_format=[ImpliedMathFormat.Yes],
    register_format_hint=[None],
    enable_direct_indexing=[True],
    enable_dest_direct_addressing=[False, True],
    transpose=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32, 512],
    is_perf=[True],
)
def test_perf_matmul_dest_di_grouped_quasar(
    perf_report,
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    dimensions,
    format,
    implied_math_format,
    register_format_hint,
    enable_direct_indexing,
    enable_dest_direct_addressing,
    transpose,
    run_types,
    loop_factor,
    is_perf,
):
    run_matmul(
        math_fidelity=math_fidelity,
        dest_sync_mode=dest_sync_mode,
        dest_acc=dest_acc,
        dimensions=dimensions,
        format=format,
        implied_math_format=implied_math_format,
        register_format_hint=register_format_hint,
        enable_direct_indexing=enable_direct_indexing,
        enable_dest_direct_addressing=enable_dest_direct_addressing,
        transpose=transpose,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
