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
        MathOperation.Reciprocal,
        MathOperation.Sqrt,
        MathOperation.Rsqrt,
        MathOperation.Silu,
        MathOperation.Gelu,
        MathOperation.Exp,
        MathOperation.TopKLocalSort,
        MathOperation.TopKMerge,
        MathOperation.TopKRebuild,
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
    ],  # Number of SFPU iterations
    fast_mode=[
        FastMode.Yes,
        FastMode.No,
    ],
    stable_sort=[
        StableSort.Yes,
        StableSort.No,
    ],
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],  # Specifying different input sizes to cover different tile counts
)
def test_perf_eltwise_unary_sfpu(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    fast_mode,
    stable_sort,
    input_dimensions,
    workers_tensix_coordinates,
):
    # Skip tests where template parameters are not used by the operation
    # Operations that don't use is_fp32_dest_acc_en parameter
    ops_without_dest_acc = {
        MathOperation.Abs,
        MathOperation.Acosh,
        MathOperation.Asinh,
        MathOperation.Celu,
        MathOperation.Cos,
        MathOperation.Elu,
        MathOperation.Exp2,
        MathOperation.Exp,
        MathOperation.Fill,
        MathOperation.Gelu,
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

    # Operations that use FAST_MODE
    ops_with_fast_mode = {
        MathOperation.Exp,
        MathOperation.Rsqrt,
        MathOperation.Sqrt,
    }

    # Operations that use STABLE_SORT
    ops_with_stable_sort = {
        MathOperation.TopKLocalSort,
        MathOperation.TopKMerge,
        MathOperation.TopKRebuild,
    }

    # Skip if dest_acc varies but operation doesn't use it
    if mathop in ops_without_dest_acc and dest_acc == DestAccumulation.Yes:
        pytest.skip(f"{mathop} does not use dest_acc parameter - skipping Yes variant")

    # Skip if fast_mode varies but operation doesn't use it
    if mathop not in ops_with_fast_mode and fast_mode == FastMode.Yes:
        pytest.skip(f"{mathop} does not use fast_mode parameter - skipping Yes variant")

    # Skip if stable_sort varies but operation doesn't use it
    if mathop not in ops_with_stable_sort and stable_sort == StableSort.Yes:
        pytest.skip(
            f"{mathop} does not use stable_sort parameter - skipping Yes variant"
        )

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
            MATH_OP(mathop=mathop),
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

    configuration.run(perf_report, location=workers_tensix_coordinates)
