# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE perf probe for the Wormhole `erfinv` kernel (round-3 cleanup).

erfinv maps 1:1 to `ckernel_sfpu_erfinv.h`, which calls `sfpu_sqrt_custom`
(`ckernel_sfpu_sqrt_custom.h`) twice. The branch reduces that helper's
Newton-Raphson refinement from 2 -> 1 iteration for the erfinv call sites, so a
main-vs-branch run isolates the cost of the dropped NR step.

Two states are measured:
  * Float16_b, dest_acc=No   -> the headline bf16 sweep format.
  * Float32,   dest_acc=Yes  -> fp32 dest-accumulate path.

Only PerfRunType.MATH_ISOLATE is requested: cycles/tile lands in the TILE_LOOP
row as mean(MATH_ISOLATE) and the math ELF size is TEXT_SIZE(MATH_ISOLATE).
"""

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
    CLAMP_NEGATIVE,
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

_OPS = [MathOperation.Erfinv]


def _run(formats, mathop, dest_acc, input_dimensions):
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    configuration = PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FAST_MODE(FastMode.No),
            STABLE_SORT(StableSort.No),
            CLAMP_NEGATIVE(False),
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
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )
    return configuration


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=_OPS,
    input_dimensions=[[128, 64]],
)
def test_perf_erfinv_bf16(perf_report, formats, mathop, input_dimensions):
    _run(formats, mathop, DestAccumulation.No, input_dimensions).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float32]),
    mathop=_OPS,
    input_dimensions=[[128, 64]],
)
def test_perf_erfinv_fp32(perf_report, formats, mathop, input_dimensions):
    _run(formats, mathop, DestAccumulation.Yes, input_dimensions).run(perf_report)
