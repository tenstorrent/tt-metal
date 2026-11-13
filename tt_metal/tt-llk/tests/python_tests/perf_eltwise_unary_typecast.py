# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
On-silicon perf benchmark for the SFPU typecast op (issue #46751).

Measures cycles/tile for the typecast variants whose SFPLOADMACRO fast path was
re-introduced for Blackhole, plus the plain-loop 8/16-bit-output arms
(_TYPECAST_PLAIN_LOOP_PERF_CASES). The macro fast path in the calculate_typecast_*
primitives is gated `#ifndef DISABLE_SFPLOADMACRO`; compiling with
TT_METAL_DISABLE_SFPLOADMACRO=1 selects the plain-loop fallback.
Running this module twice -- once without the env (macro ON,
optimized) and once with TT_METAL_DISABLE_SFPLOADMACRO=1 (macro OFF, baseline)
-- gives a clean A/B on the same tree.

Unlike test_perf_eltwise_unary_sfpu (which dispatches a single MathOperation),
typecast is selected by the (IN, OUT) DataFormat pair via typecast_tile<IN, OUT>
/ the shared SfpuType::typecast dispatch. Functional and performance tests use
the same sources/eltwise_unary_typecast_test.cpp kernel.
"""

import pytest
from helpers.llk_params import PerfRunType
from helpers.param_config import parametrize
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.tile_constants import MAX_FACE_R_DIM, MAX_NUM_FACES
from test_eltwise_unary_typecast import TYPECAST_SWEEP, _build_typecast_config


@pytest.mark.perf
@parametrize(
    **TYPECAST_SWEEP,
    loop_factor=[
        16,
    ],  # Number of iterations to run the test in order to minimize profiler overhead in measurement
    iterations=[
        8,
    ],  # SFPU iteration count; the typecast dispatch hardcodes 8 ITERATIONS internally
)
def test_perf_eltwise_unary_typecast(
    perf_report,
    formats,
    dest_acc,
    approx_mode,
    loop_factor,
    iterations,
    input_dimensions,
):
    # Calculate tile count from input dimensions
    tile_count_A, tile_count_B, _ = calculate_tile_and_face_counts(
        input_dimensions,
        input_dimensions,
        face_r_dim=MAX_FACE_R_DIM,
        num_faces=MAX_NUM_FACES,
    )

    configuration = _build_typecast_config(
        formats,
        dest_acc,
        approx_mode,
        input_dimensions,
        tile_count_A,
        tile_count_B,
        is_perf=True,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.MATH_ISOLATE,
        ],
        loop_factor=loop_factor,
        iterations=iterations,
    )

    configuration.run(perf_report)
