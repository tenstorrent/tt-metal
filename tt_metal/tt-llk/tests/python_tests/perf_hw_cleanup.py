# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole compute_kernel_hw_cleanup (sources/hw_cleanup_perf.cpp): one identity datacopy per iteration,
with the three-thread cleanup and the re-init it forces (PERF_STAGE 1) or without (PERF_STAGE 0); unit: one iteration.
"""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    PERF_STAGE,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b


@pytest.mark.perf
@parametrize(stage=[0, 1])
def test_perf_hw_cleanup(perf_report, stage):
    # parametrize hands a single axis as a one-element tuple
    if isinstance(stage, tuple):
        (stage,) = stage
    configuration = PerfConfig(
        "sources/hw_cleanup_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[PerfRunType.L1_TO_L1],
        templates=[PERF_STAGE(stage)],
        runtimes=[TILE_COUNT(1), NUM_FACES(4), LOOP_FACTOR(16)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=4,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
