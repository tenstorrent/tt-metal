# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Mega SFPU accuracy sweep.

Runs a deterministic ramp sweep for every transcendental SFPU op across the
supported unary input/output format combinations and hardware configs, and
writes one shard CSV per variant. accuracy/conftest.py merges current-run
shards into one CSV per op. Sanity-assert only (no ULP threshold gating).
"""

from itertools import product

import pytest
from conftest import skip_for_coverage
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
)
from helpers.param_config import input_output_formats
from helpers.test_config import TestConfig

# Transcendental / approximated ops where HW accuracy is interesting.
TRANSCENDENTAL_OPS = [
    MathOperation.Exp,
    MathOperation.Exp2,
    MathOperation.Log,
    MathOperation.Log1p,
    MathOperation.Sqrt,
    MathOperation.Rsqrt,
    MathOperation.Reciprocal,
    MathOperation.Sin,
    MathOperation.Cos,
    MathOperation.Tanh,
    MathOperation.Gelu,
    MathOperation.Silu,
    MathOperation.Elu,
    MathOperation.Celu,
    MathOperation.Atanh,
    MathOperation.Asinh,
    MathOperation.Acosh,
    MathOperation.Hardsigmoid,
    MathOperation.Erfinv,
]

# Ops that support FastMode.
SUPPORTED_FAST_MODE_OPS = [
    MathOperation.Log1p,
    MathOperation.Exp,
    MathOperation.Rsqrt,
    MathOperation.Sqrt,
]

FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16,
        DataFormat.Float16_b,
    ]
)

# (formats, approx, op, fast, dest) — fast varies only for fast-mode ops.
ACCURACY_PARAMS = list(
    (fmt, approx, op, fast, dest)
    for fmt, approx, op, dest in product(
        FORMATS,
        [ApproximationMode.No, ApproximationMode.Yes],
        TRANSCENDENTAL_OPS,
        [DestAccumulation.No, DestAccumulation.Yes],
    )
    for fast in (
        [FastMode.No, FastMode.Yes] if op in SUPPORTED_FAST_MODE_OPS else [FastMode.No]
    )
)


@skip_for_coverage
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "formats,approx_mode,mathop,fast_mode,dest_acc", ACCURACY_PARAMS
)
def test_sfpu_accuracy_sweep(
    formats: InputOutputFormat,
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    fast_mode: FastMode,
    dest_acc: DestAccumulation,
):
    if mathop == MathOperation.Tanh and approx_mode == ApproximationMode.Yes:
        pytest.skip(reason="Metal tanh does not support approximation mode")

    if (
        dest_acc == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and (
            formats.input_format == DataFormat.Float16
            or formats == InputOutputFormat(DataFormat.Float32, DataFormat.Float16)
        )
    ):
        pytest.skip(reason="This combination is not supported on BH architecture")

    from accuracy.accuracy_harness import run_case

    run_case(mathop, formats, approx_mode, fast_mode, dest_acc)
