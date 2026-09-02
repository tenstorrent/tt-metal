# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Perf slice for the kernels touched by ldjurovic/vif_optimizations_wh.

perf_eltwise_unary_sfpu.py already measures most of these, but only on
Float16_b; `calculate_comp_int` is reachable only with an Int32 input and
therefore has no coverage there at all. Rather than widen the shared sweep (its
format matrix is asserted against PERF_SWEEP_OPS), this module carries just the
affected ops, at the same loop_factor/iterations as the main sweep so numbers
are directly comparable.

Run it once per header variant with the ELF cache wiped in between —
/tmp/tt-llk-build keys on source path, not header content, so a stale cache
silently measures the previous variant.
"""

import pytest
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
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
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    STABLE_SORT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

_DIMS = [[128, 64]]  # tile_cnt: 8, same as the main perf sweep

# Float kernels the branch rewrites (metal llk_sfpu/*.h).
_FLOAT_OPS = [
    MathOperation.Sign,
    MathOperation.Heaviside,
    MathOperation.Hardshrink,
    MathOperation.UnaryEq,
    MathOperation.UnaryNe,
    MathOperation.UnaryGt,
    MathOperation.UnaryLt,
    MathOperation.UnaryGe,
    MathOperation.UnaryLe,
]

# calculate_comp_int: reached only when the runtime math_format is Int32.
_INT_COMP_OPS = [
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
]


def _config(
    formats,
    mathop,
    dest_acc,
    unpack_to_dest,
    input_dimensions,
    source="sources/eltwise_unary_sfpu_perf.cpp",
):
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    return PerfConfig(
        source,
        formats,
        run_types=ALL_PERF_RUN_TYPES,
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


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=_FLOAT_OPS,
    input_dimensions=_DIMS,
)
def test_perf_vif_float(perf_report, formats, mathop, input_dimensions):
    _config(formats, mathop, DestAccumulation.No, False, input_dimensions).run(
        perf_report
    )


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    mathop=_INT_COMP_OPS,
    input_dimensions=_DIMS,
)
def test_perf_vif_comp_int32(perf_report, formats, mathop, input_dimensions):
    # Int32 unpacks straight into a 32-bit DEST, mirroring the correctness path.
    _config(formats, mathop, DestAccumulation.Yes, True, input_dimensions).run(
        perf_report
    )


# The metal relu_min / relu_max need their own sources: sfpu_operations.h routes
# SfpuType::relu_min/relu_max to tt-llk's own _relu_min_/_relu_max_, so the shared
# op would measure a different kernel. sources/vif_relu_*_perf.cpp are copies of
# eltwise_unary_sfpu_perf.cpp with only the SFPU call swapped, so the measurement
# path (and therefore the MATH_ISOLATE number) is directly comparable.
@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=[MathOperation.ReluMin, MathOperation.ReluMax],
    input_dimensions=_DIMS,
)
def test_perf_vif_metal_relu(perf_report, formats, mathop, input_dimensions):
    source = (
        "sources/vif_relu_min_perf.cpp"
        if mathop == MathOperation.ReluMin
        else "sources/vif_relu_max_perf.cpp"
    )
    _config(
        formats,
        mathop,
        DestAccumulation.No,
        False,
        input_dimensions,
        source=source,
    ).run(perf_report)
