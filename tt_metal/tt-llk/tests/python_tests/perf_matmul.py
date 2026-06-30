# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
from helpers.format_config import DataFormat, FormatConfig, is_32b_dest_needed
from helpers.llk_params import Fp32DestMode, MathFidelity, PerfRunType, Transpose
from helpers.matmul_sweep import (
    generate_matmul_dimension_combinations,
    generate_tile_dims,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)

# Important K dimensions to test
KT_DIMS = [1, 2, 3, 4, 8, 32]


def matmul_combos(
    formats: List[FormatConfig],
    is_32b_dest_en: List[Fp32DestMode],
):
    def _dest_bank_max_tiles(format: FormatConfig, is_32b_dest_en: Fp32DestMode):
        if is_32b_dest_needed(format) or is_32b_dest_en == Fp32DestMode.Yes:
            return 4
        return 8

    unique_max_tiles = set(
        _dest_bank_max_tiles(fmt, acc) for fmt in formats for acc in is_32b_dest_en
    )
    dimensions = {
        max_tiles: generate_matmul_dimension_combinations(max_tiles, kt_dims=KT_DIMS)
        for max_tiles in unique_max_tiles
    }

    return [
        (format, accumulation, dims)
        for format in formats
        for accumulation in is_32b_dest_en
        for dims in dimensions[_dest_bank_max_tiles(format, accumulation)]
    ]


@pytest.mark.perf
@parametrize(
    combos=matmul_combos(
        formats=input_output_formats(
            [
                DataFormat.Float16_b,
                DataFormat.Float16,
                DataFormat.Float32,
                DataFormat.Bfp8_b,
            ]
        ),
        is_32b_dest_en=[Fp32DestMode.No, Fp32DestMode.Yes],
    ),
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_perf_matmul(
    perf_report,
    combos,
    math_fidelity,
):

    formats, is_32b_dest_en, (matrix_a, matrix_b) = combos

    if is_32b_dest_needed(formats) and is_32b_dest_en == Fp32DestMode.No:
        pytest.skip("Dest accumulation must be enabled for this format")

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    # Calculate all matmul dimensions using helper function
    dims = generate_tile_dims((matrix_a, matrix_b))

    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim

    configuration = PerfConfig(
        "sources/matmul_perf.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            THROTTLE_LEVEL(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            NUM_FACES(),
            LOOP_FACTOR(16),
            TILE_COUNT(variant_tile_count),
            CRK_TILE_DIMM(dims.ct_dim, dims.rt_dim, dims.kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=variant_tile_count,
            tile_count_B=variant_tile_count,
            tile_count_res=variant_tile_count,
        ),
        is_32b_dest_en=is_32b_dest_en,
    )

    configuration.run(perf_report)
