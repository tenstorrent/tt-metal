# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Binary SFPU ADD/SUB/RSUB with DstRoundingMode::NearestEven.

This is the arm binary_ng runs for every bf16 ADD/SUB/RSUB it sends to the SFPU: the
fp32 result is narrowed into the bf16 Dest with a software round-to-nearest-even. It
is only meaningful with a bf16 Dest (dest_acc=No); with fp32 accumulation the rounding
is skipped and the kernel is the Default one perf_eltwise_binary_sfpu already measures.

It lives in its own module, and so in its own perf catalog entry, because the perf
gate keys a point on its non-null sweep columns: recording ``dst_rounding`` in
perf_eltwise_binary_sfpu would re-key every point of that module and detach it from
its baseline.
"""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    DstRoundingMode,
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
    SFPU_DST_ROUNDING_MODE,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwrsub,
    ],
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_eltwise_binary_sfpu_rne(
    perf_report,
    formats,
    mathop,
    loop_factor,
    iterations,
    input_dimensions,
):
    dest_acc = DestAccumulation.No
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
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(iterations),
            SFPU_DST_ROUNDING_MODE(DstRoundingMode.NearestEven),
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
