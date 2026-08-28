# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness for the grouped (multi-replay) dest-addressed DI path.

The dest-addressed DI stream covers `group_tiles` output tiles per replay. When a block needs more
than one group, the executor positions the dest base with TT_SETRWC(..., SET_D) between replays and
the recording's dest offsets are relative to the group. That path is exercised only by blocks
larger than one group, and `exact_dest_fill` in the shared sweep does not generate the shapes where
it matters most, so those shapes are listed explicitly here.

The `rt > ct >= 2` shapes are the ones whose baseline pays two dest re-base instructions per column
(14 overhead instructions vs 6). They are also the shapes that depend on TT_SETRWC(..., SET_D)
setting the dest counter to an absolute row value: if it incremented instead, every group after the
first would write to the wrong dest tiles, silently.
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    PerfRunType,
    Transpose,
)
from helpers.param_config import parametrize
from quasar.test_matmul_quasar import test_matmul as run_matmul

# (rt, ct, kt) -> (input_A_dimensions, input_B_dimensions), tiles = 32 datums.
# rt*ct = 8 tiles throughout, which needs a 16-bit dest (DestAccumulation.No) at DestSync.Half.
#   4x2 : rt > ct >= 2  -> baseline pays the dest re-base fixups; grouped DI drops them
#   2x4 : reuse_a, same tile count, no fixups -> isolates the fixup contribution
#   8x1 : tall but ct == 1 -> no fixups either, two groups per column
GROUPED_DIMENSIONS = [
    ([128, 32], [32, 64]),  # rt=4 ct=2 kt=1
    ([128, 128], [128, 64]),  # rt=4 ct=2 kt=4
    ([64, 32], [32, 128]),  # rt=2 ct=4 kt=1
    ([64, 128], [128, 128]),  # rt=2 ct=4 kt=4
    ([256, 32], [32, 32]),  # rt=8 ct=1 kt=1
    ([256, 128], [128, 32]),  # rt=8 ct=1 kt=4
]


@pytest.mark.quasar
@parametrize(
    format=[InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)],
    # LoFi: group_tiles=4. HiFi2: group_tiles=2, so twice as many groups. HiFi4: group_tiles=1 and
    # the block is multi-tile, so the op falls back -- a control that must match the baseline.
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4],
    dest_sync_mode=[DestSync.Half],
    dest_acc=[DestAccumulation.No],
    dimensions=GROUPED_DIMENSIONS,
    implied_math_format=[ImpliedMathFormat.Yes],
    register_format_hint=[None],
    enable_direct_indexing=[True],
    enable_dest_direct_addressing=[False, True],
    transpose=[Transpose.No],
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_matmul_dest_di_grouped(
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
    )
