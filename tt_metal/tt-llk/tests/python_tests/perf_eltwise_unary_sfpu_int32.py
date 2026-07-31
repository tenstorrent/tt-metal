# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


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
    CLAMP_NEGATIVE,
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

# int32 unary ops share the eltwise_unary_sfpu_perf.cpp dispatch but need an
# Int32 format, so they run through a dedicated MATH_ISOLATE test below.
#
# Coverage note: AddInt32/SubInt32 (binop_with_unary.h) currently have perf-only
# coverage here and no functional golden/assert, because the int32-unary
# functional sweep is blocked by the fast-tilize gap (tt-llk #495). Their integer
# core is exercised functionally via the binary path (_add_int_/_sub_int_ in
# test_sfpu_binary.py, SfpuElwadd), but the unary calculate_add_int32/
# calculate_sub_int32 wrappers themselves stay perf-only until #495 is resolved.
_INT32_UNARY_OPS = [
    MathOperation.AddInt32,
    MathOperation.SubInt32,
    MathOperation.AbsInt32,
    MathOperation.ReluMin,
    MathOperation.BitwiseNot,
    MathOperation.LogicalNot,
    MathOperation.Fill,
]


def _run_math_isolate(formats, mathop, input_dimensions):
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    unpack_to_dest = formats.input_format.is_32_bit()

    return PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FAST_MODE(FastMode.No),
            STABLE_SORT(StableSort.No),
            CLAMP_NEGATIVE(False),
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
        dest_acc=DestAccumulation.No,
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    mathop=_INT32_UNARY_OPS,
    input_dimensions=[[128, 64]],
)
def test_perf_eltwise_unary_sfpu_int32(perf_report, formats, mathop, input_dimensions):
    _run_math_isolate(formats, mathop, input_dimensions).run(perf_report)
