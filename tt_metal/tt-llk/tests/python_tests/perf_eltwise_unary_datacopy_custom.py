# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole copy_tile_custom path (sources/eltwise_unary_datacopy_custom_perf.cpp); unit: one 32x32 tile,
eight per DEST section."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b
TILES_PER_SECTION = 8


@pytest.mark.perf
@parametrize(num_tiles=[8, 32])
def test_perf_eltwise_unary_datacopy_custom(perf_report, num_tiles):
    # parametrize hands a single axis as a one-element tuple
    if isinstance(num_tiles, tuple):
        (num_tiles,) = num_tiles
    configuration = PerfConfig(
        "sources/eltwise_unary_datacopy_custom_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        runtimes=[
            TILE_COUNT(num_tiles),
            NUM_FACES(4),
            NUM_BLOCKS(num_tiles // TILES_PER_SECTION),
            NUM_TILES_IN_BLOCK(TILES_PER_SECTION),
            LOOP_FACTOR(64),
        ],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=num_tiles,
            tile_count_B=1,
            tile_count_res=num_tiles,
            num_faces=4,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
