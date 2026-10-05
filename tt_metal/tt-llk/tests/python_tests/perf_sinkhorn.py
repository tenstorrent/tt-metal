# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole SFPU sinkhorn_4x4 (sources/sinkhorn_perf.cpp) in its documented flow: copy, exp init, row-max
subtraction, exponential, sinkhorn_4x4. PERF_STAGE 0 copies and packs alone, 1 adds the exponential stage, 2 the sinkhorn
body with SINKHORN_ITERS passes; unit: one 32x32 tile.
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
    PERF_STAGE,
    SINKHORN_ITERS,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b


@pytest.mark.perf
@parametrize(variant=[(0, 20), (1, 20), (2, 1), (2, 5), (2, 20)])
def test_perf_sinkhorn(perf_report, variant):
    # parametrize hands a single axis as a one-element tuple
    if len(variant) == 1:
        (variant,) = variant
    stage, iters = variant
    configuration = PerfConfig(
        "sources/sinkhorn_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
        templates=[PERF_STAGE(stage), SINKHORN_ITERS(iters)],
        runtimes=[TILE_COUNT(2), LOOP_FACTOR(16)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=2,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
