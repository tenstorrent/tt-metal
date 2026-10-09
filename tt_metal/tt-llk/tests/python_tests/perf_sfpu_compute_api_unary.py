# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The unary SFPU registry bodies as their <op>_tile issues them, four 8-row calls per tile (VectorMode::RC, 8
iterations), on Float16_b with a 16-bit DEST; perf_eltwise_unary_sfpu.py times them as one 32-row call per tile.
"""

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    FusedSort,
    MathOperation,
    StableSort,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.sfpu_domains import sfpu_unary_ops
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
    STABLE_SORT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

# The TopK entry points make one call per tile, as the registry sweep does, and the registry already issues erfinv
# as four 8-row calls.
_NOT_IN_API_FORM = {
    MathOperation.TopKDefuse,
    MathOperation.TopKLocalSort,
    MathOperation.TopKMerge,
    MathOperation.TopKRebuild,
    MathOperation.Erfinv,
}

_OPS = sorted(set(sfpu_unary_ops()) - _NOT_IN_API_FORM, key=lambda op: op.name)


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    approx_mode=[ApproximationMode.Yes, ApproximationMode.No],
    mathop=_OPS,
)
def test_perf_sfpu_compute_api_unary(perf_report, formats, approx_mode, mathop):
    input_dimensions = [128, 64]
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_compute_api_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(8),
            FAST_MODE(FastMode.No),
            STABLE_SORT(StableSort.No),
            FUSED_SORT(FusedSort.No),
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
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
