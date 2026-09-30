# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.constraints import distinct_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
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
    # TestConfig promotes dest_acc=No to Yes for outlier format combos, so asking
    # for both would record two rows with an identical key (the same kernel twice).
    return distinct_dest_accumulation_modes(
        formats, [DestAccumulation.Yes, DestAccumulation.No]
    )


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
    dest_acc=lambda formats: get_dest_accum_modes(formats),
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
        run_types=ALL_PERF_RUN_TYPES,
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
        run_types=ALL_PERF_RUN_TYPES,
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
        run_types=ALL_PERF_RUN_TYPES,
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


# The binary SFPU ops the tests above do not time. Every op here has a functional test
# in test_eltwise_binary_sfpu.py; this is its cost, so a kernel change can be measured
# before and after (the LLK SFPU report, tt-llk/sfpu_report, measures from these). The
# sweep is deliberately narrow -- the formats a reviewer reads, and approx No, which is
# the only mode test_eltwise_binary_sfpu compiles -- so it adds little to the nightly.
_EXTENDED_FLOAT_OPS = [
    MathOperation.SfpuXlogy,
    MathOperation.SfpuLogaddexp,
    MathOperation.SfpuLogaddexp2,
    MathOperation.SfpuAtan2,
    MathOperation.SfpuBinaryMax,
    MathOperation.SfpuBinaryMin,
    MathOperation.SfpuBinaryFmod,
    MathOperation.SfpuBinaryRemainder,
    MathOperation.SfpuElwLt,
    MathOperation.SfpuElwGt,
    MathOperation.SfpuElwLe,
    MathOperation.SfpuElwGe,
    MathOperation.SfpuElwEq,
    MathOperation.SfpuElwNe,
    MathOperation.SfpuIsclose,
    MathOperation.SfpuLogsigmoid,
]

_EXTENDED_INT32_OPS = [
    MathOperation.SfpuDivInt32,
    MathOperation.SfpuDivInt32Floor,
    MathOperation.SfpuRemainderInt32,
    MathOperation.SfpuFmodInt32,
    MathOperation.SfpuRsubInt32,
    MathOperation.SfpuMulInt32,
    MathOperation.SfpuGcd,
    MathOperation.SfpuLcm,
    MathOperation.SfpuMaxInt32,
    MathOperation.SfpuMinInt32,
    MathOperation.SfpuEqInt,
    MathOperation.SfpuNeInt,
    MathOperation.SfpuBitwiseAnd,
    MathOperation.SfpuBitwiseOr,
    MathOperation.SfpuBitwiseXor,
    MathOperation.SfpuElwLt,
    MathOperation.SfpuElwGt,
    MathOperation.SfpuElwLe,
    MathOperation.SfpuElwGe,
]

_EXTENDED_UINT32_OPS = [
    MathOperation.SfpuMaxUint32,
    MathOperation.SfpuMinUint32,
    MathOperation.SfpuRemainderUint32,
]


def _run_binary_perf(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
    unpack_to_dest,
):
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
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
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32],
        same=True,
    ),
    approx_mode=[ApproximationMode.No],
    mathop=_EXTENDED_FLOAT_OPS,
    dest_acc=lambda formats: get_dest_accum_modes(formats),
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_eltwise_binary_sfpu_float_extended(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    _run_binary_perf(
        perf_report,
        formats,
        mathop,
        approx_mode,
        dest_acc,
        loop_factor,
        iterations,
        input_dimensions,
        unpack_to_dest=formats.input_format.is_32_bit()
        and dest_acc == DestAccumulation.No,
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32, DataFormat.UInt32], same=True),
    approx_mode=[ApproximationMode.No],
    mathop=_EXTENDED_INT32_OPS + _EXTENDED_UINT32_OPS,
    # The integer kernels are functionally tested with a 32-bit Dest only.
    dest_acc=[DestAccumulation.Yes],
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_eltwise_binary_sfpu_int_extended(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    unsigned = formats.input_format == DataFormat.UInt32
    if unsigned != (mathop in _EXTENDED_UINT32_OPS):
        pytest.skip(f"{mathop.name} is not a {formats.input_format.name} op")
    if get_chip_architecture() == ChipArchitecture.BLACKHOLE and mathop in (
        MathOperation.SfpuDivInt32,
        MathOperation.SfpuDivInt32Floor,
    ):
        # Pre-existing: with TT_LLK_DISABLE_ASSERTS=1, which every perf run sets, the
        # BH div_int32 kernels do not compile ("too few lregs to hold live values");
        # test_eltwise_binary_sfpu.py fails the same way under that flag.
        pytest.skip(
            "BH div_int32 does not compile with TT_LLK_DISABLE_ASSERTS=1; see tt-metal#58598"
        )
    if (
        get_chip_architecture() == ChipArchitecture.BLACKHOLE
        and mathop == MathOperation.SfpuLcm
    ):
        pytest.skip(
            "SfpuLcm dest_acc=Yes is codegen-sensitive and hangs on Blackhole; see tt-metal#52997"
        )
    _run_binary_perf(
        perf_report,
        formats,
        mathop,
        approx_mode,
        dest_acc,
        loop_factor,
        iterations,
        input_dimensions,
        unpack_to_dest=True,
    )
