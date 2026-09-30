# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
On-silicon perf benchmark for the SFPU quantization kernels (quant, requant, dequant) with a per-tensor scale.

Two LLK forms of the scale are measured side by side on sources/sfpu_quant_scalar_perf.cpp:
  * ``tile``: the scale is a DEST tile, copied into DEST next to the input for every output tile (an
    unpack-to-DEST handshake per tile) and loaded by the body once per row of 32 datums. This is the form the
    binary_ng scalar kernel ran for a per-tensor scale.
  * ``scalar``: the scale is loaded into the SFPU once by the init, together with the zero point; only the input
    tile is copied into DEST and the body has no scale load.

Formats follow the ttnn ops: quant Float32 -> Int32, requant Int32 -> Int32, dequant Int32 -> Float32, all in a
32-bit Dest with the 32-bit inputs unpacked straight to Dest.
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    QUANT_SCALAR_CFG,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

_QUANT_FORMATS = {
    "quant": InputOutputFormat(DataFormat.Float32, DataFormat.Int32),
    "requant": InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
    "dequant": InputOutputFormat(DataFormat.Int32, DataFormat.Float32),
}


@pytest.mark.perf
@parametrize(
    quant_op=["quant", "requant", "dequant"],
    scale_form=["tile", "scalar"],
    loop_factor=[16],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_quant_scalar(perf_report, quant_op, scale_form, loop_factor, input_dimensions):
    formats = _QUANT_FORMATS[quant_op]

    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_quant_scalar_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[
            QUANT_SCALAR_CFG(quant_op=quant_op, scale_form=scale_form),
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
        # The 32-bit inputs unpack straight into the 32-bit Dest, as the quant ops run in ttnn.
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
    )

    configuration.run(perf_report)
