# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE (and full run-type) perf for the ternary SFPU kernels that lack
dedicated coverage. The addc kernels are flagged in the PR #48696 verification
(table 3); lerp/snake_beta are the other standalone ternary-dispatch SFPU ops
with no perf test of their own:

  addcmul    -> metal llk_sfpu/ckernel_sfpu_addcmul.h     (out = a + value*b*c)
  addcdiv    -> metal llk_sfpu/ckernel_sfpu_addcdiv.h     (out = a + value*b/c)
  lerp       -> metal llk_sfpu/ckernel_sfpu_lerp.h        (out = a + c*(b - a))
  snake_beta -> metal llk_sfpu/ckernel_sfpu_snake_beta.h  (out = x + sin(alpha*x)^2/beta)

These are ternary (3 Dest tiles, plus a scalar for the addc kernels) and do not
fit the unary/binary harnesses, so they run through a dedicated ternary source.
The cycles/tile number lands in the TILE_LOOP row of the .post.csv as
mean(MATH_ISOLATE), and the math ELF size is TEXT_SIZE(MATH_ISOLATE).
"""

import struct

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    ITERATIONS,
    LOOP_FACTOR,
    NUM_FACES,
    SFPU_TERNARY_OP,
    SFPU_TERNARY_SCALAR,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

_SCALAR_VALUE_BITS = struct.unpack("<I", struct.pack("<f", 2.0))[0]


def _run(formats, mathop, dest_acc, loop_factor, iterations, input_dimensions):
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_ternary_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        # SPEED_OF_LIGHT-style: everything is a compile-time template so the measured
        # kernel has no runtime-parameter reads. These sweep values are single-valued,
        # so making them templates does not expand the build matrix.
        templates=[
            SFPU_TERNARY_OP(mathop),
            SFPU_TERNARY_SCALAR(_SCALAR_VALUE_BITS),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(iterations),
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
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )
    return configuration


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ],
        same=True,
    ),
    dest_acc=[
        DestAccumulation.Yes,
        DestAccumulation.No,
    ],
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_addcmul(
    perf_report,
    formats,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    _run(
        formats,
        MathOperation.SfpuAddcmul,
        dest_acc,
        loop_factor,
        iterations,
        input_dimensions,
    ).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
        ],
        same=True,
    ),
    dest_acc=[
        DestAccumulation.Yes,
        DestAccumulation.No,
    ],
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_addcdiv(
    perf_report,
    formats,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    _run(
        formats,
        MathOperation.SfpuAddcdiv,
        dest_acc,
        loop_factor,
        iterations,
        input_dimensions,
    ).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
        ],
        same=True,
    ),
    dest_acc=[
        DestAccumulation.Yes,
        DestAccumulation.No,
    ],
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_lerp(
    perf_report,
    formats,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    _run(
        formats,
        MathOperation.SfpuLerp,
        dest_acc,
        loop_factor,
        iterations,
        input_dimensions,
    ).run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
        ],
        same=True,
    ),
    dest_acc=[
        DestAccumulation.Yes,
        DestAccumulation.No,
    ],
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_snake_beta(
    perf_report,
    formats,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    _run(
        formats,
        MathOperation.SfpuSnakeBeta,
        dest_acc,
        loop_factor,
        iterations,
        input_dimensions,
    ).run(perf_report)
