# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Ternary SFPU perf, sharing sources/sfpu_ternary_test.cpp with the functional tests.

LOOP_FACTOR is 16 here and 1 in the functional tests. The .post.csv divides by
loop_factor * tile_cnt, so the two stay comparable. Both tensor shapes run here;
the functional tests stay at [64, 64] so their ULP key keeps its old ceiling.
"""

import pytest
import test_sfpu_ternary as _func
from conftest import blackhole_only
from helpers.constraints import skip_if_32bit_input_without_dest_acc
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES

_PERF_INPUT_DIMENSIONS = [[64, 64], [128, 64]]


def _perf_formats(mathop):
    formats = [DataFormat.Float16_b, DataFormat.Float32]
    if mathop == MathOperation.SfpuAddcmul:
        formats.append(DataFormat.Bfp8_b)
    # where is the only ternary op whose functional tests cover Int32.
    if mathop == MathOperation.SfpuWhere:
        formats.append(DataFormat.Int32)
    return input_output_formats(formats, same=True)


@pytest.mark.perf
@parametrize(
    mathop=[
        MathOperation.SfpuAddcmul,
        MathOperation.SfpuAddcdiv,
        MathOperation.SfpuLerp,
        MathOperation.SfpuSnakeBeta,
        MathOperation.SfpuWhere,
    ],
    formats=_perf_formats,
    dest_acc=[
        DestAccumulation.Yes,
        DestAccumulation.No,
    ],
    loop_factor=[16],
    input_dimensions=_PERF_INPUT_DIMENSIONS,
)
def test_perf_sfpu_ternary(
    perf_report,
    mathop,
    formats,
    dest_acc,
    loop_factor,
    input_dimensions,
):
    if mathop == MathOperation.SfpuWhere:
        _func._skip_where_config(formats, dest_acc)
    else:
        skip_if_32bit_input_without_dest_acc(formats, dest_acc)
    _func._run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        input_dimensions=input_dimensions,
        loop_factor=loop_factor,
        approx_mode=ApproximationMode.No,
        is_perf=True,
        perf_report=perf_report,
        run_types=ALL_PERF_RUN_TYPES,
    )


# Same Blackhole regression as test_sfpu_ternary_approx_after_lreg_clobber.
@blackhole_only
@pytest.mark.perf
@parametrize(
    mathop=_func._TERNARY_RECIP_NEWTON_OPS,
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    loop_factor=[16],
    input_dimensions=_PERF_INPUT_DIMENSIONS,
)
def test_perf_sfpu_ternary_approx_after_lreg_clobber(
    perf_report,
    mathop,
    formats,
    dest_acc,
    loop_factor,
    input_dimensions,
):
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)
    _func._run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        input_dimensions=input_dimensions,
        loop_factor=loop_factor,
        approx_mode=ApproximationMode.Yes,
        clobber_lregs=True,
        is_perf=True,
        perf_report=perf_report,
        run_types=ALL_PERF_RUN_TYPES,
    )
