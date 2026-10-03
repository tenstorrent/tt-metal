# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole pack untilize init forms after a one-tile matmul (sources/matmul_pack_untilize_perf.cpp): the
init once, pack_untilize_dest_init per block, custom_pack_untilize_dest_init per block; unit: one block of one tile."""

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
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    PACK_UNTILIZE_INIT,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b
BLOCKS = 8


@pytest.mark.perf
@parametrize(init=["once", "standard", "custom"])
def test_perf_matmul_pack_untilize(perf_report, init):
    if isinstance(init, tuple):  # parametrize hands a single axis as a one-element tuple
        (init,) = init
    configuration = PerfConfig(
        "sources/matmul_pack_untilize_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.PACK_ISOLATE],
        templates=[MATH_FIDELITY(MathFidelity.LoFi), PACK_UNTILIZE_INIT(init)],
        runtimes=[
            NUM_BLOCKS(BLOCKS),
            NUM_TILES_IN_BLOCK(1),
            TILE_COUNT(BLOCKS),
            LOOP_FACTOR(32),
        ],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=BLOCKS,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
