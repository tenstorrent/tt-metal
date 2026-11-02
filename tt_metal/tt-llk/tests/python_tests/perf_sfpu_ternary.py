# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE (and full run-type) perf for the ternary SFPU kernels that lack
dedicated coverage. The addc kernels are flagged in the PR #48696 verification
(table 3); lerp/snake_beta are the other standalone ternary-dispatch SFPU ops
with no perf test of their own:

  addcmul    -> metal llk_sfpu/ckernel_sfpu_addcmul.h     (out = a + value*b*c)
  addcdiv    -> metal llk_sfpu/ckernel_sfpu_addcdiv.h     (out = a + value*b/c)
  lerp       -> metal llk_sfpu/ckernel_sfpu_lerp.h        (out = a + c*(b - a))
  snake_beta -> metal llk_sfpu/ckernel_sfpu_snake_beta.h  (out = x + sin(alpha*x)^2/beta)
  where      -> metal llk_sfpu/ckernel_sfpu_where.h         (out = a != 0 ? b : c)

These are ternary (3 Dest tiles, plus a scalar for the addc kernels) and do not
fit the unary/binary harnesses, so they run through a dedicated ternary source.
The cycles/tile number lands in the TILE_LOOP row of the .post.csv as
mean(MATH_ISOLATE), and the math ELF size is TEXT_SIZE(MATH_ISOLATE).
"""

import pytest
import test_sfpu_ternary as _func
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES


@pytest.mark.perf
@parametrize(
    mathop=[
        MathOperation.SfpuAddcmul,
        MathOperation.SfpuAddcdiv,
        MathOperation.SfpuLerp,
        MathOperation.SfpuSnakeBeta,
        MathOperation.SfpuWhere,
    ],
    formats=lambda mathop: input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32]
        + ([DataFormat.Bfp8_b] if mathop == MathOperation.SfpuAddcmul else []),
        same=True,
    ),
    dest_acc=[
        DestAccumulation.Yes,
        DestAccumulation.No,
    ],
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_ternary(
    perf_report,
    mathop,
    formats,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    _func._run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        input_dimensions=input_dimensions,
        loop_factor=loop_factor,
        iterations=iterations,
        approx_mode=ApproximationMode.No,
        is_perf=True,
        perf_report=perf_report,
        run_types=ALL_PERF_RUN_TYPES,
    )
