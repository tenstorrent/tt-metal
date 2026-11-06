# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE perf for the EMA SFPU kernel (ckernel_sfpu_ema.h).

The EMA kernel had no perf coverage at all, which is why the Wormhole scheduling
change to _compute_ema_math_ could not be quantified from the existing suite. This
drives sources/sfpu_ema_perf.cpp, whose TILE_LOOP marker under MATH_ISOLATE covers the
math pipe with no dest handshake with pack.

It does still include the datacopy that feeds Dest in the 16-bit forms: it retires the SrcA valid bits unpack sets,
so mean(MATH_ISOLATE) is the SFPU block plus a fixed datacopy that cancels in a delta, not the kernel alone.

cycles/tile lands in the TILE_LOOP row of the .post.csv as mean(MATH_ISOLATE).

Both DEST widths are swept; the Float32 form unpacks straight to Dest and has no datacopy.
"""


import pytest
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig, PerfRunType
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    EMA_ALPHA_BETA,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    fp32_bits,
)

# Same smoothing factor as the functional test, so the perf and correctness runs
# exercise the identical constant pair.
EMA_ALPHA = 0.25
EMA_BETA = 1.0 - EMA_ALPHA


@pytest.mark.perf
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    ],
    dest_acc=get_valid_dest_accumulation_modes,
    loop_factor=[16],  # amortise profiler overhead
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_ema(
    perf_report,
    formats,
    dest_acc,
    loop_factor,
    input_dimensions,
):
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_ema_perf.cpp",
        formats,
        run_types=[
            PerfRunType.MATH_ISOLATE,
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        # Everything compile-time so the measured kernel does no runtime-parameter
        # reads; all sweep values are single-valued so this does not expand the matrix.
        templates=[
            APPROX_MODE(ApproximationMode.No),
            EMA_ALPHA_BETA(
                alpha_bits=fp32_bits(EMA_ALPHA), beta_bits=fp32_bits(EMA_BETA)
            ),
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
