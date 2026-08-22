# Lane EU binarypow perf module — split from perf_eltwise_binary_sfpu per the
# PerfSchemaError one-schema-per-file rule (FRESH_CPP_IMPL parametrization emits
# a distinct column set; stacking it into the shared binary module contaminated
# every op riding that module — pin-18 e2e2 weekly RED, adjudicated).
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


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
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    FRESH_CPP_IMPL,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)


# Lane EU coverage expansion (binarypow-fresh row): fresh semantic pow
# (impl 1) vs the byte-untouched calculate_sfpu_binary_pow hand kernel
# (impl 3 — the production POW dispatch routes through calculate_sfpu_binary,
# so the metal__ckernel_sfpu_binary_pow entry had zero nodes before this
# selector).  MATH_ISOLATE only, all-new node ids.
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=[MathOperation.SfpuElwpow],
    fresh_cpp_impl=[3, 1],
)
def test_perf_fresh_cpp_binary_pow(perf_report, formats, mathop, fresh_cpp_impl):
    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        [128, 64], [128, 64], face_r_dim=16, num_faces=4
    )
    configuration = PerfConfig(
        "sources/eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FRESH_CPP_IMPL(fresh_cpp_impl),
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
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


# Lane DH fitted-kernel placeholder (tt-polynomial-fitter frontier atan winner
# + the storm-S1 quadrant fixup; provenance in fresh_cpp/atan2_fitted.h).
# Dedicated perf family, all-new node ids: impl 2 = fitted body (sem arm),
# impl 0 = production hand kernel (hand arm); MATH_ISOLATE only (mirrors
# test_perf_fresh_cpp_binary_float_s1's node conventions).
