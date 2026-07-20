# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE perf for the Wormhole comparison-to-zero SFPU kernels converted raw
TTI -> pure sfpi on branch `ldjurovic/expand_on_spfi_update`. Each op below maps to
one changed function in llk_sfpu/ckernel_sfpu_comp.h so a main-vs-branch run isolates
the cost of the conversion:

  eqz/nez/ltz/gtz/lez/gez (float)  -> calculate_comp        (float, bitwise-magnitude)
  eqz/nez (uint16)                 -> calculate_comp_uint16  (DataLayout::U16)
  eqz    (uint32)                  -> calculate_eqz_uint32   (DataLayout::U32)
  nez    (uint32)                  -> calculate_nez_uint32   (DataLayout::U32)

Only PerfRunType.MATH_ISOLATE is requested to keep the sweep small; the cycles/tile
number lands in the TILE_LOOP row of the .post.csv as mean(MATH_ISOLATE), and the
math ELF size is TEXT_SIZE(MATH_ISOLATE).
"""

import pytest
from conftest import skip_for_blackhole
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

# Float calculate_comp covers all six comparison-to-zero modes.
_FLOAT_COMP_OPS = [
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
]

# uint16 (calculate_comp_uint16) and uint32 (calculate_eqz/nez_uint32) only support
# equality against zero.
_UINT_COMP_OPS = [
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
]


def _run(formats, mathop, input_dimensions):
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    unpack_to_dest = formats.input_format.is_32_bit()

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
        dest_acc=DestAccumulation.No,
    )
    return configuration


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=_FLOAT_COMP_OPS,
    input_dimensions=[[128, 64]],
)
def test_perf_sfpu_comp_float(perf_report, formats, mathop, input_dimensions):
    _run(formats, mathop, input_dimensions).run(perf_report)


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.UInt16], same=True),
    mathop=_UINT_COMP_OPS,
    input_dimensions=[[128, 64]],
)
def test_perf_sfpu_comp_uint16(perf_report, formats, mathop, input_dimensions):
    _run(formats, mathop, input_dimensions).run(perf_report)


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.UInt32], same=True),
    mathop=_UINT_COMP_OPS,
    input_dimensions=[[128, 64]],
)
def test_perf_sfpu_comp_uint32(perf_report, formats, mathop, input_dimensions):
    _run(formats, mathop, input_dimensions).run(perf_report)
