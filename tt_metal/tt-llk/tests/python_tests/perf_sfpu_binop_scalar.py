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
a dedicated source that calls calculate_binop_with_scalar directly. The
cycles/tile number lands in the TILE_LOOP row of the .post.csv as
mean(MATH_ISOLATE), and the math ELF size is TEXT_SIZE(MATH_ISOLATE).
"""

import struct

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    ITERATIONS,
    LOOP_FACTOR,
    NUM_FACES,
    SFPU_BINOP_MODE,
    SFPU_UNARY_SCALAR,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

_SCALAR_BITS = struct.unpack("<I", struct.pack("<f", 2.0))[0]

_RUN_TYPES = [
    PerfRunType.L1_TO_L1,
    PerfRunType.UNPACK_ISOLATE,
    PerfRunType.MATH_ISOLATE,
    PerfRunType.PACK_ISOLATE,
    PerfRunType.L1_CONGESTION,
]


def _run(formats, mathop, dest_acc, loop_factor, iterations, input_dimensions):
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_binop_scalar_perf.cpp",
        formats,
        run_types=_RUN_TYPES,
        templates=[
            SFPU_BINOP_MODE(mathop),
            SFPU_UNARY_SCALAR(_SCALAR_BITS),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(iterations),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )
    return configuration


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
        ],
        same=True,
    ),
    dest_acc=[
        DestAccumulation.Yes,
        DestAccumulation.No,
    ],
    mathop=[
        MathOperation.ScalarAdd,
        MathOperation.ScalarSub,
        MathOperation.ScalarMul,
        MathOperation.ScalarDiv,
        MathOperation.ScalarRsub,
    ],
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_binop_scalar(
    perf_report,
    formats,
    dest_acc,
    mathop,
    loop_factor,
    iterations,
    input_dimensions,
):
    _run(
        formats,
        mathop,
        dest_acc,
        loop_factor,
        iterations,
        input_dimensions,
    ).run(perf_report)
