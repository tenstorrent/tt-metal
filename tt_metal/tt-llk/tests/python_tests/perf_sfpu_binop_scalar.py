# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE (and full run-type) perf for the float unary-with-scalar binops in
binop_with_unary.h. These were flagged in the PR #48696 verification (table 3)
as having no MathOperation / golden and no functional or perf coverage:

  add  -> calculate_binop_with_scalar<ADD>    (out = x + s)
  sub  -> calculate_binop_with_scalar<SUB>    (out = x - s)
  mul  -> calculate_binop_with_scalar<MUL>    (out = x * s)
  div  -> calculate_binop_with_scalar<DIV>    (out = x * (1/d))
  rsub -> calculate_binop_with_scalar<RSUB>   (out = s - x)

They are unary-with-scalar (one Dest tile + an fp32 scalar), so they run through
a dedicated shared functional/perf source that calls
calculate_binop_with_scalar directly. The
cycles/tile number lands in the TILE_LOOP row of the .post.csv as
mean(MATH_ISOLATE), and the math ELF size is TEXT_SIZE(MATH_ISOLATE).
"""

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_sfpu_binop_scalar import (
    SFPU_BINOP_SCALAR_SWEEP,
    _run_sfpu_binop_scalar,
)


@pytest.mark.perf
@parametrize(
    **SFPU_BINOP_SCALAR_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
    is_perf=[True],
)
def test_perf_sfpu_binop_scalar(
    perf_report,
    formats,
    dest_acc,
    mathop,
    loop_factor,
    iterations,
    input_dimensions,
    run_types,
    is_perf,
):
    _run_sfpu_binop_scalar(
        formats,
        dest_acc,
        mathop,
        input_dimensions=input_dimensions,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        iterations=iterations,
    )
