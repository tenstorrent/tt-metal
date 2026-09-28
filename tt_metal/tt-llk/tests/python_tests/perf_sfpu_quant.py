# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
On-silicon perf benchmark for the SFPU quant family (quant / requant / dequant).

The quant kernels have no ckernel::BinaryOp, so this module uses the dedicated
sources/sfpu_quant_perf.cpp (structurally sources/eltwise_binary_sfpu_perf.cpp) with the
kernel selected by SFPU_QUANT_VARIANT: one row per compute-API entry point of
tt_metal/hw/inc/api/compute/quantization.h. Operands travel as raw Int32 words in a 32-bit
Dest, exactly as ttnn's binary_ng quantize ops drive them.
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    LOOP_FACTOR,
    NUM_FACES,
    QUANT_VARIANTS,
    SFPU_QUANT_VARIANT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    ZERO_POINT,
)

# fp32 bit pattern of the zero-point 5.0; the value does not affect the instruction stream.
_ZERO_POINT_BITS = 0x40A00000


@pytest.mark.perf
@parametrize(
    quant_variant=list(QUANT_VARIANTS),
    approx_mode=[ApproximationMode.No],
    loop_factor=[
        16,
    ],  # Number of iterations to run the test in order to minimize profiler overhead in measurement
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],
)
def test_perf_sfpu_quant(
    perf_report,
    quant_variant,
    approx_mode,
    loop_factor,
    input_dimensions,
):
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    dest_acc = DestAccumulation.Yes  # every quant kernel loads/stores 32-bit Dest words

    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_quant_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            SFPU_QUANT_VARIANT(quant_variant),
            APPROX_MODE(approx_mode),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            ZERO_POINT(_ZERO_POINT_BITS),
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
