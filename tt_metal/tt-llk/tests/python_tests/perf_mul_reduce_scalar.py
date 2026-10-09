# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole mul_reduce_scalar row (mul_reduce_scalar_tile, sources/mul_reduce_scalar_perf.cpp); unit: one
input tile of the row."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    ROW_TILES,
    TILE_COUNT,
)
from helpers.tile_shape import construct_tile_shape

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

# (fidelity, tiles per row, row length compiled in: 0 for the runtime count, tile rows). The last three are the DeepSeek
# RMSNorm's rows at LoFi: one and three 16x32 tiles (512 and 1536 wide) and seven 32x32 tiles (7168 wide).
VARIANTS = [
    (fidelity, num_tiles, 0, 32)
    for num_tiles in (1, 2, 4, 8)
    for fidelity in (MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4)
] + [
    (MathFidelity.LoFi, 1, 1, 16),
    (MathFidelity.LoFi, 3, 3, 16),
    (MathFidelity.LoFi, 7, 7, 32),
]


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_mul_reduce_scalar(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    fidelity, num_tiles, row_tiles, tile_rows = variant
    tile_dimensions = [tile_rows, 32]
    tile_shape = construct_tile_shape(tile_dimensions)
    configuration = PerfConfig(
        "sources/mul_reduce_scalar_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[MATH_FIDELITY(fidelity), ROW_TILES(row_tiles)],
        runtimes=[
            TILE_COUNT(num_tiles),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim, tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim, tile_shape.num_faces_c_dim),
            LOOP_FACTOR(64),
        ],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=num_tiles,
            tile_count_B=num_tiles,
            tile_count_res=1,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
            sfpu=False,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
