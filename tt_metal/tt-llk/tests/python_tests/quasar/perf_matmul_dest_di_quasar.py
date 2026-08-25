# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused DI vs dest-addressed-DI perf comparison.

Deliberately tiny: the dest-addressed DI stream only fits the 64-entry math replay buffer
for ct_dim * rt_dim * fidelity_phases * 16 <= 64, so LoFi 4-tile blocks are where it can
remove the most (per-tile MOP restart + per-block dest bookkeeping). HiFi4 is the control:
its stream never fits, the path falls back to the plain DI MOP, and the two rows must match.

Read mean(MATH_ISOLATE) in the .post.csv -- that is cycles per tile-pass with the unpack
bottleneck removed, which is where a math-thread instruction saving is visible at all.
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
from quasar.test_matmul_quasar import matmul_dimensions
from quasar.test_matmul_quasar import test_matmul as run_matmul


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    format=[InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)],
    # LoFi: 4 tiles * 1 phase * 16 ops = 64, fits exactly. HiFi4: 64 ops per tile, falls back.
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    dest_sync_mode=[DestSync.Half],
    # 32-bit dest caps the block at 4 tiles, which is the largest stream that fits at LoFi.
    dest_acc=[DestAccumulation.Yes],
    dimensions=lambda dest_acc, dest_sync_mode: matmul_dimensions(
        dest_acc,
        dest_sync_mode,
        exact_dest_fill=True,
        is_perf=True,
    ),
    implied_math_format=[ImpliedMathFormat.Yes],
    register_format_hint=[None],
    enable_direct_indexing=[True],
    enable_dest_direct_addressing=[False, True],
    transpose=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    # Dest-addressed DI trades a bigger one-time replay load (64 instruction writes vs 15)
    # for a cheaper loop, so the verdict depends on how many tile-passes amortize the init.
    # Sweeping loop_factor separates the fixed per-kernel cost from the per-tile saving.
    loop_factor=[32, 128, 512],
    is_perf=[True],
)
def test_perf_matmul_dest_di_quasar(
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
