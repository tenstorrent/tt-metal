# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE perf for the Wormhole complex-bucket SFPU kernels touched on branch
`ldjurovic/wh_sfpu_cleanup4`. Each op maps 1:1 to one changed kernel header so a
main-vs-branch run isolates the cost of the change (see
SFPU_PERF_COMPARISON_GUIDE.md):

  silu       -> sfpu/ckernel_sfpu_silu.h            (#pragma GCC unroll 8)
  tanhshrink -> llk_sfpu/ckernel_sfpu_tanhshrink.h  (fp32 large-|x| recip 2->1 NR)
  floor      -> sfpu/ckernel_sfpu_rounding_ops.h    (#pragma GCC unroll 8)
  ceil       -> sfpu/ckernel_sfpu_rounding_ops.h    (#pragma GCC unroll 8)
  trunc      -> sfpu/ckernel_sfpu_rounding_ops.h    (#pragma GCC unroll 8)
  frac       -> sfpu/ckernel_sfpu_rounding_ops.h    (#pragma GCC unroll 8)

Two states are measured per op:
  * Float16_b, dest_acc=No   -> the headline bf16 sweep format.
  * Float32,   dest_acc=Yes  -> exercises the fp32 dest-accumulate path, which is
                                the only path that reaches the tanhshrink NR change.

Only PerfRunType.MATH_ISOLATE is requested: cycles/tile lands in the TILE_LOOP row
as mean(MATH_ISOLATE) and the math ELF size is TEXT_SIZE(MATH_ISOLATE).
"""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
    PerfRunType,
    StableSort,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    FAST_MODE,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    STABLE_SORT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

_OPS = [
    MathOperation.Silu,
    MathOperation.Tanhshrink,
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
    MathOperation.Frac,
]


def _run(formats, mathop, dest_acc, input_dimensions):
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    configuration = PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FAST_MODE(FastMode.No),
            STABLE_SORT(StableSort.No),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
            LOOP_FACTOR(16),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count_A,
            tile_count_B=tile_count_B,
            tile_count_res=tile_count_A,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )
    return configuration


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=_OPS,
    input_dimensions=[[128, 64]],
)
def test_perf_complex_cleanup_bf16(perf_report, formats, mathop, input_dimensions):
    _run(formats, mathop, DestAccumulation.No, input_dimensions).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float32]),
    mathop=_OPS,
    input_dimensions=[[128, 64]],
)
def test_perf_complex_cleanup_fp32(perf_report, formats, mathop, input_dimensions):
    _run(formats, mathop, DestAccumulation.Yes, input_dimensions).run(perf_report)
