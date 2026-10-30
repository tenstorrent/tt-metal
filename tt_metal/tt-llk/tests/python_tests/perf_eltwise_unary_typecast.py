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
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    PerfRunType,
)
from helpers.param_config import parametrize
from helpers.stimuli_generator import calculate_tile_and_face_counts
from test_eltwise_unary_typecast import _build_typecast_config

# The (IN, OUT, dest_acc) typecast cases that exercise the SFPLOADMACRO path,
# including the paths re-introduced by issue #46751.
#
# dest_acc is the production setting EXCEPT for the two rows marked below: their
# macro fast path is gated !DST_ACCUM_MODE, but ttnn.typecast forces dest_acc=Yes
# for 32-bit outputs (typecast.cpp:38-41), so production takes the plain loop. Those
# rows measure the macro in isolation and their speedup is not production-realized.
_TYPECAST_PERF_CASES = [
    # Float16_b -> UInt16: routes through calculate_typecast_fp32_to_uint16 (the
    # Compute API maps Float16_b-in-Dest -> UInt16 to it, no FP32 load from L1).
    (DataFormat.Float16_b, DataFormat.UInt16, DestAccumulation.No),
    # uint16_to_fp32 -- NOT production-reachable (Float32 out -> dest_acc=Yes).
    (DataFormat.UInt16, DataFormat.Float32, DestAccumulation.No),
    # uint32_to_fp16b, macro mode-agnostic; UInt32 needs 32-bit Dest.
    (DataFormat.UInt32, DataFormat.Float16_b, DestAccumulation.Yes),
    # Exact uint32 -> fp32 conversion with a three-cycle SFPLOADMACRO pipeline.
    (DataFormat.UInt32, DataFormat.Float32, DestAccumulation.Yes),
    # uint16_to_uint32 -- NOT production-reachable (UInt32 out -> dest_acc=Yes).
    (DataFormat.UInt16, DataFormat.UInt32, DestAccumulation.No),
    # int32_to_uint16, macro mode-agnostic; Int32 needs 32-bit Dest.
    (DataFormat.Int32, DataFormat.UInt16, DestAccumulation.Yes),
]

# The plain-loop 8/16-bit-output typecasts, one row per calculate_typecast_* input, at dest_acc=Yes:
# ttnn.typecast forces a 32-bit Dest for 32-bit and UInt8 endpoints (preserve_fp32_precision).
_TYPECAST_PLAIN_LOOP_PERF_CASES = [
    (DataFormat.Float32, DataFormat.UInt8, DestAccumulation.Yes),
    (DataFormat.Int32, DataFormat.UInt8, DestAccumulation.Yes),
    (DataFormat.UInt32, DataFormat.UInt8, DestAccumulation.Yes),
    (DataFormat.UInt16, DataFormat.UInt8, DestAccumulation.Yes),
    (DataFormat.UInt32, DataFormat.UInt16, DestAccumulation.Yes),
    (DataFormat.UInt8, DataFormat.UInt16, DestAccumulation.Yes),
]


@pytest.mark.perf
@parametrize(
    typecast_case=_TYPECAST_PERF_CASES + _TYPECAST_PLAIN_LOOP_PERF_CASES,
    approx_mode=[ApproximationMode.No],
    loop_factor=[
        16,
    ],  # Number of iterations to run the test in order to minimize profiler overhead in measurement
    iterations=[
        8,
    ],  # SFPU iteration count; the typecast dispatch hardcodes 8 ITERATIONS internally
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],
)
def test_perf_eltwise_unary_typecast(
    perf_report,
    typecast_case,
    approx_mode,
    loop_factor,
    iterations,
    input_dimensions,
):
    input_format, output_format, dest_acc = typecast_case
    formats = InputOutputFormat(input_format, output_format)

    # Calculate tile count from input dimensions
    tile_count_A, tile_count_B, _ = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = _build_typecast_config(
        formats,
        dest_acc,
        approx_mode,
        input_dimensions,
        None,
        tile_count_A,
        None,
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
