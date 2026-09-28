# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
On-silicon perf benchmark for the integer unary SFPU shift kernels (left_shift / right_shift).

The registry-driven sweep in perf_eltwise_unary_sfpu.py covers the float unary ops; the
integer shift ops have no registry entry (their correctness harness is
test_eltwise_unary_sfpu_int / _int_shift), so this module measures them on their production
configuration: Int32 in a 32-bit Dest, unpacked straight to Dest, with the compile-time shift
amount SFPU_SHIFT_AMOUNT. The two shift arms are separate kernels
(ckernel_sfpu_unary_shift.h: calculate_left_shift / calculate_right_shift), so each is a row.
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    FusedSort,
    MathOperation,
    PerfRunType,
    StableSort,
    Transpose,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    FUSED_SORT,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    SFPU_SHIFT_AMOUNT,
    STABLE_SORT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

_INT_SHIFT_OPS = [MathOperation.LeftShift, MathOperation.RightShift]


@pytest.mark.perf
@parametrize(
    mathop=_INT_SHIFT_OPS,
    shift_amount=[3],  # the instruction stream does not depend on the amount
    loop_factor=[
        16,
    ],  # Number of iterations to run the test in order to minimize profiler overhead in measurement
    iterations=[
        32,
    ],  # Number of SFPU iterations
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],
)
def test_perf_eltwise_unary_sfpu_int(
    perf_report,
    mathop,
    shift_amount,
    loop_factor,
    iterations,
    input_dimensions,
):
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    dest_acc = DestAccumulation.Yes  # the 32-bit int path needs a 32-bit Dest

    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(iterations),
            FAST_MODE(FastMode.No),
            STABLE_SORT(StableSort.No),
            FUSED_SORT(FusedSort.No),
            CLAMP_NEGATIVE(False),
            SFPU_SHIFT_AMOUNT(shift_amount),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
            LOOP_FACTOR(loop_factor),
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
            twos_complement=True,
        ),
        # Int32 has no SrcA/SrcB unpack path: unpack straight into the 32-bit Dest.
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
