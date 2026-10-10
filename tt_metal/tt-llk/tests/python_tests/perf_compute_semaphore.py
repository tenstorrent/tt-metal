# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the compute semaphore ring on a datacopy pipeline (sources/compute_semaphore_perf.cpp); unit: one tile.
A ring depth below the section's eight tiles deadlocks: the unpack stops before the math has a whole section to hand on.
"""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, SEMAPHORE_RING, TILE_COUNT

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b
TILES = 8


@pytest.mark.perf
@parametrize(ring_depth=[0, 15, 8])
def test_perf_compute_semaphore(perf_report, ring_depth):
    # parametrize hands a single axis as a one-element tuple
    if isinstance(ring_depth, tuple):
        (ring_depth,) = ring_depth
    assert (
        ring_depth == 0 or ring_depth >= TILES
    ), "a ring below one DEST section deadlocks"
    configuration = PerfConfig(
        "sources/compute_semaphore_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[SEMAPHORE_RING(ring_depth)],
        runtimes=[TILE_COUNT(TILES), LOOP_FACTOR(64)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=TILES,
            tile_count_B=1,
            tile_count_res=TILES,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
