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

_OPS_WITHOUT_DEST_ACC = {
    MathOperation.Abs,
    # Acosh/Asinh now select their log1p polynomial precision from the dest-accum
    # (is_fp32_dest_acc_en) flag, so both modes are exercised.
    MathOperation.Celu,
    MathOperation.Cos,
    MathOperation.Elu,
    MathOperation.Exp2,
    MathOperation.Exp,
    MathOperation.Fill,
    MathOperation.Gelu,
    MathOperation.GeluTanh,
    MathOperation.Hardsigmoid,
    MathOperation.Log,
    MathOperation.Neg,
    MathOperation.Silu,
    MathOperation.Sin,
    MathOperation.Square,
    MathOperation.Threshold,
    MathOperation.ReluMax,
    MathOperation.ReluMin,
}

_OPS_WITH_FAST_MODE = {
    MathOperation.Exp,
    MathOperation.Rsqrt,
    MathOperation.Sqrt,
}

_OPS_WITH_STABLE_SORT = {
    MathOperation.TopKLocalSort,
    MathOperation.TopKMerge,
    MathOperation.TopKRebuild,
}


def _get_dest_acc_modes(mathop):
    if mathop in _OPS_WITHOUT_DEST_ACC:
        return [DestAccumulation.No]
    return [DestAccumulation.Yes, DestAccumulation.No]


def _get_fast_modes(mathop):
    if mathop in _OPS_WITH_FAST_MODE:
        return [FastMode.Yes, FastMode.No]
    return [FastMode.No]


def _get_stable_sort_modes(mathop):
    if mathop in _OPS_WITH_STABLE_SORT:
        return [StableSort.Yes, StableSort.No]
    return [StableSort.No]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    approx_mode=[
        ApproximationMode.Yes,
        ApproximationMode.No,
    ],
    math_op=[
        MathOperation.Reciprocal,
        MathOperation.Sqrt,
        MathOperation.Rsqrt,
        MathOperation.Silu,
        MathOperation.Gelu,
        MathOperation.GeluTanh,
        MathOperation.Exp,
        MathOperation.Lrelu,
        MathOperation.ReluMin,
        MathOperation.Erfinv,
        MathOperation.Heaviside,
        MathOperation.Softshrink,
        MathOperation.Softsign,
        MathOperation.Log,
        MathOperation.TopKLocalSort,
        MathOperation.TopKMerge,
        MathOperation.TopKRebuild,
    ],
    dest_acc=lambda math_op: _get_dest_acc_modes(math_op),
    loop_factor=[
        16,
    ],  # Number of iterations to run the test in order to minimize profiler overhead in measurement
    iterations=[
        32,
    ],  # Number of SFPU iterations
    fast_mode=lambda math_op: _get_fast_modes(math_op),
    stable_sort=lambda math_op: _get_stable_sort_modes(math_op),
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],  # Specifying different input sizes to cover different tile counts
)
def test_perf_eltwise_unary_sfpu_float(
    perf_report,
    formats,
    math_op,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    fast_mode,
    stable_sort,
    input_dimensions,
):
    # Calculate tile count from input dimensions
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    # If dest_acc is on, we unpack Float32 into 16-bit format in src registers
    # (later copied over in dest reg for SFPU op)
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    configuration = PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=math_op),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
            FAST_MODE(fast_mode),
            STABLE_SORT(stable_sort),
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
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)


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
    MathOperation.ReluMin,
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
            MATH_OP(mathop=math_op),
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
        dest_acc=DestAccumulation.No,
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    approx_mode=[ApproximationMode.No],
    math_op=_INT32_UNARY_OPS,
    fast_mode=[FastMode.No],
    dest_acc=[DestAccumulation.Yes],
    input_dimensions=[[128, 64]],
)
def test_perf_eltwise_unary_sfpu_int(
    perf_report,
    formats,
    approx_mode,
    math_op,
    fast_mode,
    dest_acc,
    input_dimensions,
):
    _run_math_isolate(formats, math_op, input_dimensions).run(perf_report)
