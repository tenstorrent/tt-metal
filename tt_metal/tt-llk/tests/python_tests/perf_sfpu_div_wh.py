# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE perf probe for the dedicated binary DIV kernel
(calculate_sfpu_binary_div) on branch `ldjurovic/wh_sfpu_cleanup4`.

The shared binary harness now routes SfpuElwdiv to the production
calculate_sfpu_binary_div (see helpers/include/sfpu_operations.h), so this
probe measures the real kernel. Two states:
  Float16_b / acc=No  (bf16 path: sfpu_reciprocal_iter + bf16 RNE)
  Float32   / acc=Yes (fp32 path: sfpu_reciprocal_iter + Markstein residual)
"""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)


def _run(formats, dest_acc, input_dimensions):
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=MathOperation.SfpuElwdiv),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )
    return configuration


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    input_dimensions=[[128, 64]],
)
def test_perf_div_bf16(perf_report, formats, input_dimensions):
    _run(formats, DestAccumulation.No, input_dimensions).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float32]),
    input_dimensions=[[128, 64]],
)
def test_perf_div_fp32(perf_report, formats, input_dimensions):
    _run(formats, DestAccumulation.Yes, input_dimensions).run(perf_report)
