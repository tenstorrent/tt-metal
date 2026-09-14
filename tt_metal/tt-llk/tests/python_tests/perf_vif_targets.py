# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Perf for `calculate_comp_int`, which the shared sweep structurally cannot reach.

`perf_eltwise_unary_sfpu.py` asserts its format matrix against `PERF_SWEEP_OPS`
and does not carry Int32, but `calculate_comp_int` is reachable only with an
Int32 input — so these six comparison-to-zero modes have no perf coverage
anywhere. This module is the extra slice that supplies it, exactly as
`test_perf_eltwise_unary_sfpu_comp_uint16` and `_comp_uint32` already do for
their formats: a slice that bypasses `PERF_SWEEP_OPS` rather than widening it.

Nothing here duplicates the shared sweep. Every float op these kernels touch is
already in `_OP_DOMAIN_REGISTRY`, hence in `PERF_SWEEP_OPS`, and measured at
these same parameters; `run_llk_perf_wormhole.sh` collects the whole directory,
so carrying them here too would measure those rows twice in every perf shard.

loop_factor/iterations/dimensions match the shared sweep so the numbers stay
directly comparable with it.

When A/B-ing header variants, wipe the ELF cache between runs.
`TestConfig.variant_id` hashes the `-I` include-directory paths, not header
*content*, and `build_elfs()` skips outright once `.build_complete` is set, so
an unwiped $RUNNER_TEMP/tt-llk-build silently replays the previous variant's
ELFs and reports a 1.00x delta that means nothing.
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

# calculate_comp_int: reached only when the runtime math_format is Int32.
_INT_COMP_OPS = [
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
]


def _config(formats, mathop, dest_acc, unpack_to_dest, input_dimensions):
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    return PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
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
    formats=input_output_formats([DataFormat.Int32], same=True),
    mathop=_INT_COMP_OPS,
    input_dimensions=_DIMS,
)
def test_perf_vif_comp_int32(perf_report, formats, mathop, input_dimensions):
    # Int32 unpacks straight into a 32-bit DEST, mirroring the correctness path.
    _config(
        formats,
        mathop,
        DestAccumulation.Yes,
        unpack_to_dest=True,
        input_dimensions=input_dimensions,
    ).run(perf_report)
