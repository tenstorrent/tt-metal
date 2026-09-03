# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Examples of focused single-op accuracy tests.

These are the "arbitrary accuracy tests" from the design: copy one, change the
op / format / config, and it contributes rows to that op's CSV via the shared
shard->merge pipeline. Sanity-assert only.
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
)


@pytest.mark.accuracy
def test_accuracy_exp_f16b_dest_acc():
    """Single-op example: exp in Float16_b with fp32 dest accumulator."""
    from accuracy.accuracy_harness import run_case

    run_case(
        MathOperation.Exp,
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        ApproximationMode.No,
        FastMode.No,
        DestAccumulation.Yes,
    )


@pytest.mark.accuracy
def test_accuracy_reciprocal_f16b():
    """Single-op example: reciprocal (two-band domain, hole around 0)."""
    from accuracy.accuracy_harness import run_case

    run_case(
        MathOperation.Reciprocal,
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        ApproximationMode.No,
        FastMode.No,
        DestAccumulation.No,
    )
