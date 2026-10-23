# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE perf for the EMA SFPU kernel (ckernel_sfpu_ema.h).

The EMA kernel had no perf coverage at all, which is why the Wormhole scheduling
change to _compute_ema_math_ could not be quantified from the existing suite. This
drives sources/sfpu_ema_perf.cpp, whose TILE_LOOP marker under MATH_ISOLATE covers the
math pipe with no dest handshake with pack.

It does still include the datacopy that feeds Dest, because that is what retires the
SrcA valid bits unpack sets -- the same arrangement eltwise_unary_sfpu_perf.cpp uses for
every unary SFPU op. So mean(MATH_ISOLATE) is not the standalone cost of the SFPU block;
it is the SFPU block plus a fixed datacopy. That offset is constant across a
before/after comparison of the kernel, so it cancels in a delta, but do not read the
absolute number as the kernel alone.

cycles/tile lands in the TILE_LOOP row of the .post.csv as mean(MATH_ISOLATE).

The kernel is hardcoded for 16-bit bf16 DEST (_ema_load/store_current_input_ use
SFPLOADI_MOD0_FLOATB at fixed fp16 offsets with no is_fp32_dest_acc_en branch), so
dest_acc=Yes is not swept -- same restriction as test_sfpu_ema.py.
"""

import struct

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    Transpose,
)
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
)

# Same smoothing factor as the functional test, so the perf and correctness runs
# exercise the identical constant pair.
EMA_ALPHA = 0.25
EMA_BETA = 1.0 - EMA_ALPHA


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.No],
    loop_factor=[16],  # amortise profiler overhead
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_ema(
    perf_report,
    dest_acc,
    loop_factor,
    input_dimensions,
):
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_ema_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE, PerfRunType.L1_TO_L1],
        # Everything compile-time so the measured kernel does no runtime-parameter
        # reads; all sweep values are single-valued so this does not expand the matrix.
        templates=[
            APPROX_MODE(ApproximationMode.No),
            EMA_ALPHA_BETA(
                alpha_bits=_f32_bits(EMA_ALPHA), beta_bits=_f32_bits(EMA_BETA)
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
        unpack_to_dest=False,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)
