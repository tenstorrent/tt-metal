# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE perf for the Welford SFPU kernel (ckernel_sfpu_welfords.h).

The Welford kernel had no perf coverage. This drives sources/sfpu_welford_perf.cpp,
whose TILE_LOOP marker under MATH_ISOLATE covers the math pipe (datacopy into dst 0 +
_calculate_welfords_tile_) with no dest handshake with pack. mean(MATH_ISOLATE) in the
TILE_LOOP row of the .post.csv is cycles per tile, including the fixed datacopy; the
datacopy_only variant measures that fixed part alone and is the flat control.

lut_size=256 is the reciprocal-LUT variant (ttnn layernorm / groupnorm);
lut_size=0 is the RISC-side 1.0f/(idx+1) fallback (ttnn welford_reduce_{w,h,hw}),
which is expected to be bound by the TRISC soft-float divide rather than the SFPU.
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
    WELFORD_PERF_CONFIG,
)

# name -> (lut_size, datacopy_only)
WELFORD_PERF_VARIANTS = {
    "lut256": (256, False),
    "lut0": (0, False),
    "datacopy_only": (256, True),
}

FORMAT_CASES = {
    "bf16_dest16": (DataFormat.Float16_b, DestAccumulation.No),
    "fp32_dest32": (DataFormat.Float32, DestAccumulation.Yes),
}


@pytest.mark.perf
@parametrize(
    variant=list(WELFORD_PERF_VARIANTS),
    format_case=list(FORMAT_CASES),
    loop_factor=[16],  # amortise profiler overhead
    input_dimensions=[[256, 32]],  # tile_cnt: 8 -> 256 samples, fits the 256-entry LUT
)
def test_perf_sfpu_welford(
    perf_report, variant, format_case, loop_factor, input_dimensions
):
    lut_size, datacopy_only = WELFORD_PERF_VARIANTS[variant]
    fmt, dest_acc = FORMAT_CASES[format_case]
    formats = InputOutputFormat(fmt, fmt)

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_welford_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE, PerfRunType.L1_TO_L1],
        templates=[
            APPROX_MODE(ApproximationMode.No),
            WELFORD_PERF_CONFIG(
                reciprocal_lut_size=lut_size, datacopy_only=datacopy_only
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
        # MATH_ISOLATE has no real unpacker, so the unpack-to-dest handshake would hang;
        # fp32 goes through SrcA here, which is fine for timing.
        unpack_to_dest=False,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)
