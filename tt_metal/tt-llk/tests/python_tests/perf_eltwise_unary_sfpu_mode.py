# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cycles per tile of each SfpuMode instruction sequence.

SfpuMode exists to trade accuracy and special-value handling for throughput, so the trade
is measured here per mode rather than inferred from instruction counts. Softsign is the
proof-of-concept op: its four modes are four distinct sequences (two reciprocal precisions
x special-value guard on/off). Measured on Blackhole p100a, bf16, MATH_ISOLATE, TILE_LOOP:
PreciseStrict 731.6, PreciseRelaxed 540.7 (the legacy sequence), ApproximateStrict 517.7,
ApproximateRelaxed 325.8 cycles/tile; the Strict guard is exactly 6 instructions (192 cycles).
"""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    FastMode,
    FusedSort,
    MathOperation,
    SfpuMode,
    StableSort,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    FUSED_SORT,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    SFPU_MODE_OVERRIDE,
    STABLE_SORT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

# Ops whose kernel API takes an SfpuMode.
SFPU_MODE_OPS = [MathOperation.Softsign]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=SFPU_MODE_OPS,
    sfpu_mode=list(SfpuMode),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_eltwise_unary_sfpu_mode(
    perf_report, formats, mathop, sfpu_mode, dest_acc, input_dimensions
):
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_OP(mathop=mathop),
            # The legacy axis the mode's precision half maps to; the kernel call itself is
            # selected by SFPU_MODE_OVERRIDE.
            APPROX_MODE(sfpu_mode.approximation_mode),
            ITERATIONS(32),
            FAST_MODE(FastMode.No),
            STABLE_SORT(StableSort.No),
            FUSED_SORT(FusedSort.No),
            CLAMP_NEGATIVE(False),
            SFPU_MODE_OVERRIDE(sfpu_mode),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
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
            tile_count_A=tile_count_A,
            tile_count_B=tile_count_B,
            tile_count_res=tile_count_A,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
