# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole block pack _llk_pack_block_contiguous_ (sources/pack_block_contiguous_perf.cpp) against the
per-tile standard _llk_pack_; unit: one tile of the given dimensions."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    L1Accumulation,
    PerfRunType,
    format_tile_sizes,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    IN_TILE_DIMS,
    L1_ACC,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PACK_BLOCK_CONTIGUOUS,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_constants import calculate_tile_size_bytes, get_tile_params

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

# (tile dimensions, tiles per call, block form)
VARIANTS = [
    (dims, num_tiles, block)
    for dims in ((32, 32), (16, 32), (1, 32))
    for num_tiles in (1, 2, 4, 8)
    for block in (True, False)
]


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_pack_block_contiguous(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    tile_dims, num_tiles, block = variant
    formats = InputOutputFormat(BF16, BF16)
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dims)
    num_faces = num_faces_r_dim * num_faces_c_dim
    configuration = PerfConfig(
        "sources/pack_block_contiguous_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.PACK_ISOLATE],
        templates=[PACK_BLOCK_CONTIGUOUS(block)],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(num_tiles),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim),
            IN_TILE_DIMS(tile_dims[0], tile_dims[1]),
            L1_ACC(L1Accumulation.No),
            NUM_BLOCKS(1),
            NUM_TILES_IN_BLOCK(num_tiles),
            LOOP_FACTOR(128),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=num_tiles,
            tile_count_B=num_tiles,
            tile_count_res=num_tiles,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=list(tile_dims),
            use_dense_tile_dimensions=True,
            operand_res_tile_size=calculate_tile_size_bytes(
                formats.output_format, list(tile_dims), format_tile_sizes
            ),
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
