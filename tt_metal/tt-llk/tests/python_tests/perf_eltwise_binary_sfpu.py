# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    FRESH_CPP_IMPL,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)


def get_dest_accum_modes(formats):
    if formats.input_format.is_32_bit() and formats.input_format.is_integer():
        return [DestAccumulation.No]
    return [DestAccumulation.Yes, DestAccumulation.No]


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
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuElwdiv,
        MathOperation.SfpuElwrsub,
        MathOperation.SfpuElwpow,
    ],
    dest_acc=[
        DestAccumulation.Yes,
        DestAccumulation.No,
    ],
    loop_factor=[
        16,
    ],  # Number of iterations to run the test in order to minimize profiler overhead in measurement
    iterations=[
        32,
    ],
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],  # Specifying different input sizes to cover different tile counts
)
def test_perf_eltwise_binary_sfpu_float(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
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
        compile_time_formats=True,
    )

    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=[MathOperation.SfpuBinaryMax, MathOperation.SfpuBinaryMin],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_binary_max_min(perf_report, formats, mathop, fresh_cpp_impl):
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


# Storm lane S1 (fresh_cpp/<op>.h semantic bodies): fresh typed-C++ selectors
# (impl 1) A/B'd against the production bodies (impl 0), MATH_ISOLATE only
# (mirrors test_perf_fresh_cpp_binary_max_min's node conventions).
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=[
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwEq,
        MathOperation.SfpuBinaryFmod,
        MathOperation.SfpuBinaryRemainder,
        MathOperation.SfpuAtan2,
    ],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_binary_float_s1(perf_report, formats, mathop, fresh_cpp_impl):
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=[MathOperation.SfpuAtan2],
    fresh_cpp_impl=[0, 2],
)
def test_perf_fitted_cpp_binary_atan2(perf_report, formats, mathop, fresh_cpp_impl):
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuBitwiseAnd],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_binary_bitwise_s1(perf_report, formats, mathop, fresh_cpp_impl):
    # Int32 raw-bit AND A/B; Int32 perf convention: dest_acc No +
    # unpack_to_dest (mirrors test_perf_fresh_cpp_add_sub_int).
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=True,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuElwadd, MathOperation.SfpuElwsub],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_add_sub_int(perf_report, formats, mathop, fresh_cpp_impl):
    # Handwritten (_add_int_/_sub_int_ SIGN_MAGNITUDE raw path) vs fresh typed-C++
    # Int32 A/B, MATH_ISOLATE only (mirrors test_perf_fresh_cpp_binary_max_min).
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        # Int32 is 32-bit integer: dest_acc No + unpack_to_dest, matching
        # test_perf_eltwise_binary_sfpu_int's derivation for these formats.
        unpack_to_dest=True,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuElwLeftShift],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_left_shift(perf_report, formats, mathop, fresh_cpp_impl):
    # Handwritten (metal ckernel_sfpu_shift.h raw-TTI fixed-LREG kernel) vs fresh
    # typed-C++ Int32 left shift A/B, MATH_ISOLATE only (mirrors
    # test_perf_fresh_cpp_add_sub_int).
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        # Int32 is 32-bit integer: dest_acc No + unpack_to_dest, matching
        # test_perf_eltwise_binary_sfpu_int's derivation for these formats.
        unpack_to_dest=True,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLogicalRightShift,
    ],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_right_shift(perf_report, formats, mathop, fresh_cpp_impl):
    # Handwritten (metal ckernel_sfpu_shift.h raw-TTI fixed-LREG kernels) vs
    # fresh typed-C++ Int32 right shift A/B, MATH_ISOLATE only (mirrors
    # test_perf_fresh_cpp_left_shift).
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        # Int32 is 32-bit integer: dest_acc No + unpack_to_dest, matching
        # test_perf_eltwise_binary_sfpu_int's derivation for these formats.
        unpack_to_dest=True,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuMulInt32],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_mul_int(perf_report, formats, mathop, fresh_cpp_impl):
    # Handwritten (metal mul_int32 SFPLOADMACRO kernel) vs fresh typed-C++ Int32
    # multiply A/B, MATH_ISOLATE only (mirrors test_perf_fresh_cpp_add_sub_int).
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        # Int32 is 32-bit integer: dest_acc No + unpack_to_dest, matching
        # test_perf_eltwise_binary_sfpu_int's derivation for these formats.
        unpack_to_dest=True,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


# Storm S2 (agent/storm-s2): canonical fresh_cpp/<op>.h semantic bodies.
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuGcd, MathOperation.SfpuDivInt32Floor],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_binary_int_storm(perf_report, formats, mathop, fresh_cpp_impl):
    # Handwritten (metal ckernel_sfpu_gcd.h / ckernel_sfpu_div_int32_floor.h) vs
    # fresh typed-C++ Int32 A/B, MATH_ISOLATE only (mirrors
    # test_perf_fresh_cpp_mul_int).
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        # Int32 is 32-bit integer: dest_acc No + unpack_to_dest, matching
        # test_perf_eltwise_binary_sfpu_int's derivation for these formats.
        unpack_to_dest=True,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuLcm],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_lcm(perf_report, formats, mathop, fresh_cpp_impl):
    # Handwritten (metal ckernel_sfpu_lcm.h raw-TTI + REPLAY binary-GCD/reciprocal
    # kernel) vs fresh typed-C++ Int32 lcm A/B, MATH_ISOLATE only (mirrors
    # test_perf_fresh_cpp_mul_int).
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        # Int32 is 32-bit integer: dest_acc No + unpack_to_dest, matching
        # test_perf_eltwise_binary_sfpu_int's derivation for these formats.
        unpack_to_dest=True,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=[MathOperation.SfpuIsclose],
    fresh_cpp_impl=[0, 1],
)
def test_perf_fresh_cpp_isclose(perf_report, formats, mathop, fresh_cpp_impl):
    # Hand-shaped production isclose (vConstIntPrgm0-parked sign mask) vs fresh
    # typed-C++ A/B, MATH_ISOLATE only (mirrors test_perf_fresh_cpp_binary_max_min).
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    approx_mode=[
        ApproximationMode.Yes,
        ApproximationMode.No,
    ],
    mathop=[
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLeftShift,
        MathOperation.SfpuElwLogicalRightShift,
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        # Included for the test-only handwritten/generated MulInt32 corpus A/B.
        MathOperation.SfpuMulInt32,
    ],
    dest_acc=lambda formats: get_dest_accum_modes(formats),
    loop_factor=[
        16,
    ],
    iterations=[
        32,
    ],
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],
)
def test_perf_eltwise_binary_sfpu_int(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
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
        compile_time_formats=True,
    )

    configuration.run(perf_report)


# Perf vehicles for the corr-only mapped corpus rows (Lane BK measurement-gap
# close): every op below already has an audited functional golden node in
# test_sfpu_binary.py and a compiled dispatch in helpers/include/
# sfpu_operations.h (call_binary_sfpu_operation), but had no perf module at
# all — the exact tier-(c) gap named per row in the Lane AZ audit. One
# representative MATH_ISOLATE node per op (the fresh_cpp_* precedent: single
# format, loop_factor 16, iterations 32, [128, 64]); correctness stays owned
# by the functional nodes, these exist so the sweep can record cycles/tile.
_EXTENDED_FLOAT_BINARY_OPS = [
    MathOperation.SfpuAtan2,
    MathOperation.SfpuBinaryFmod,
    MathOperation.SfpuBinaryRemainder,
    MathOperation.SfpuIsclose,
    MathOperation.SfpuLogsigmoid,
    # calculate_mask hard-codes its dst operands through the test adapter
    # (see test_sfpu_binary_mask); the isolate scenario measures the
    # instruction stream, which is placement-independent.
    MathOperation.SfpuMask,
    MathOperation.SfpuElwEq,
    MathOperation.SfpuElwNe,
]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=_EXTENDED_FLOAT_BINARY_OPS,
)
def test_perf_sfpu_binary_extended_float(perf_report, formats, mathop):
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


# Int32 counterparts of the corr-only rows above (bitwise, integer division,
# gcd/lcm, rsub, integer eq/ne). The functional twins run dest_acc=Yes; the
# perf convention for 32-bit integer isolates in this suite is dest_acc=No +
# unpack_to_dest (see test_perf_eltwise_binary_sfpu_int) and MATH_ISOLATE
# measures the same math instruction stream.
_EXTENDED_INT_BINARY_OPS = [
    MathOperation.SfpuBitwiseAnd,
    MathOperation.SfpuBitwiseOr,
    MathOperation.SfpuBitwiseXor,
    MathOperation.SfpuDivInt32,
    MathOperation.SfpuDivInt32Floor,
    MathOperation.SfpuGcd,
    MathOperation.SfpuLcm,
    MathOperation.SfpuRsubInt32,
    MathOperation.SfpuEqInt,
    MathOperation.SfpuNeInt,
]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    mathop=_EXTENDED_INT_BINARY_OPS,
)
def test_perf_sfpu_binary_extended_int(perf_report, formats, mathop):
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        # Int32 is 32-bit integer: dest_acc No + unpack_to_dest, matching
        # test_perf_eltwise_binary_sfpu_int's derivation for these formats.
        unpack_to_dest=True,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.UInt32,
        ],
        same=True,
    ),
    approx_mode=[
        ApproximationMode.Yes,
        ApproximationMode.No,
    ],
    mathop=[
        MathOperation.SfpuAddTopRow,
    ],
    dest_acc=lambda formats: get_dest_accum_modes(formats),
    loop_factor=[
        16,
    ],
    iterations=[
        32,
    ],
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],
)
def test_perf_eltwise_binary_sfpu_add_top_row(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    chip_arch = get_chip_architecture()

    # Skip DestAccumulation.No on Blackhole for SfpuAddTopRow
    if chip_arch == ChipArchitecture.BLACKHOLE and dest_acc == DestAccumulation.No:
        pytest.skip(
            "DestAccumulation.No is not supported for SfpuAddTopRow on Blackhole"
        )

    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.Yes:
        pytest.skip("SfpuAddTopRow does not support Float32 with DestAccumulation.Yes")

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
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
        compile_time_formats=True,
    )

    configuration.run(perf_report)
