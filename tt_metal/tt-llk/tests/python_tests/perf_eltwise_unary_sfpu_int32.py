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
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    FRESH_CPP_IMPL,
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
    # Unary shift perf vehicles (Lane BK, corr-only corpus row
    # metal__ckernel_sfpu_unary_shift): the functional golden twins are
    # test_sfpu_unary.py::test_eltwise_unary_sfpu_int[mathop:LeftShift/
    # RightShift...]; these were the only audited unary-shift kernels with no
    # perf node anywhere.
    MathOperation.LeftShift,
    MathOperation.RightShift,
]


def _run_math_isolate(formats, mathop, input_dimensions, fresh_cpp_impl=None):
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    unpack_to_dest = formats.input_format.is_32_bit()

    templates = [
        MATH_OP(mathop=mathop),
        APPROX_MODE(ApproximationMode.No),
        ITERATIONS(32),
        FAST_MODE(FastMode.No),
        STABLE_SORT(StableSort.No),
        CLAMP_NEGATIVE(False),
    ]
    # Storm lane S1: the fresh/production selector is a compile define; leave
    # it off entirely for the swept production-only nodes so their variant
    # identity is unchanged.
    if fresh_cpp_impl is not None:
        templates.append(FRESH_CPP_IMPL(fresh_cpp_impl))

    return PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=templates,
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


# Storm lane S1: fresh typed-C++ semantic selectors (impl 1) A/B'd against
# the hand-shaped production int32 bodies (impl 0) on the same node family —
# identical stimuli, marker, and metric as the swept production nodes above.
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    mathop=[MathOperation.AbsInt32, MathOperation.BitwiseNot],
    input_dimensions=[[128, 64]],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_int32(
    perf_report, formats, mathop, input_dimensions, fresh_cpp_impl
):
    _run_math_isolate(formats, mathop, input_dimensions, fresh_cpp_impl).run(
        perf_report
    )


# Storm S5: fresh/production A-B for the unaryshift row (fresh_cpp/unaryshift.h
# sem arm vs the production calculate_left_shift hand arm).  A dedicated node —
# the main int32 sweep keeps its node ids stable (no new axis there, the
# retype-tripwire lesson).
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    mathop=[MathOperation.LeftShift],
    fresh_cpp_impl=[0, 1],
)
def test_perf_unary_shift_fresh_cpp(perf_report, formats, mathop, fresh_cpp_impl):
    _run_math_isolate(formats, mathop, [128, 64], fresh_cpp_impl).run(perf_report)
