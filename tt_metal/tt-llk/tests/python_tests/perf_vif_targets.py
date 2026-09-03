# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Perf coverage for the SFPU comparison kernels the shared sweep cannot reach.

This module deliberately measures only what `perf_eltwise_unary_sfpu.py` does
not. `Sign`, `Heaviside`, `Hardshrink`, `UnaryGt`, `UnaryLt`, `UnaryGe` and
`UnaryLe` are all in `_OP_DOMAIN_REGISTRY` and therefore already in
`PERF_SWEEP_OPS` at exactly these parameters, and `run_llk_perf_wormhole.sh`
collects the whole directory -- carrying them here too would measure every one
of those rows twice in every perf shard. What is left is the genuine gap:

* `UnaryEq` / `UnaryNe` sit outside `_OP_DOMAIN_REGISTRY`
  (`helpers/sfpu_domains.py`), so they have no perf coverage anywhere.
* `calculate_comp_int` is reachable only with an Int32 input, a format the
  shared sweep's matrix does not carry, so it has none either.

Both slices follow the pattern `test_perf_eltwise_unary_sfpu_comp_uint16` and
`_comp_uint32` already use: an extra slice that bypasses `PERF_SWEEP_OPS`
rather than widening it. loop_factor/iterations/dimensions match the shared
sweep so the numbers stay directly comparable with it.

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

# calculate_unary_eq / calculate_unary_ne: the two float comparison kernels with
# no coverage in the shared sweep. The other seven touched float ops are already
# measured there at these same parameters and are not repeated here.
_FLOAT_OPS = [
    MathOperation.UnaryEq,
    MathOperation.UnaryNe,
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
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=_FLOAT_OPS,
    input_dimensions=_DIMS,
)
def test_perf_vif_float(perf_report, formats, mathop, input_dimensions):
    _config(
        formats,
        mathop,
        DestAccumulation.No,
        unpack_to_dest=False,
        input_dimensions=input_dimensions,
    ).run(perf_report)


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
