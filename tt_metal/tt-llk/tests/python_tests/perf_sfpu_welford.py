# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf for the Welford SFPU kernel (sfpu/ckernel_sfpu_welfords.h).

Drives sources/sfpu_welford_perf.cpp: one full-tile `welford_update` per input tile with the running
mean and M2 kept in the SFPU registers across tiles, the shape of the layernorm, group norm, var and
std kernels' inner loop. The rows sweep the reciprocal form (a 256-entry table, the layernorm form, or
no table, the ttnn.var / ttnn.std form), the DEST width, and a Float32 input through unpack to DEST.

MATH_ISOLATE includes the datacopy that feeds DEST, as for every unary SFPU op measured this way, so
mean(MATH_ISOLATE) is the update plus that fixed carrier.

cycles/tile lands in the TILE_LOOP row of the .post.csv as mean(MATH_ISOLATE).
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig, PerfRunType
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    WELFORD_RECIP_SIZE,
)


def _dest_acc_modes(formats):
    # A Float32 input reaches the kernel through unpack to DEST, which needs a 32-bit DEST.
    if formats.input_format.is_32_bit():
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


@pytest.mark.perf
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    ],
    dest_acc=lambda formats: _dest_acc_modes(formats),
    recip_size=[256, 0],
    loop_factor=[16],  # amortise profiler overhead
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_welford(perf_report, formats, dest_acc, recip_size, loop_factor, input_dimensions):
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_welford_perf.cpp",
        formats,
        run_types=[
            PerfRunType.MATH_ISOLATE,
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        # Everything compile-time so the measured kernel does no runtime-parameter reads.
        templates=[
            APPROX_MODE(ApproximationMode.No),
            WELFORD_RECIP_SIZE(recip_size),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        runtimes=[],
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
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)
