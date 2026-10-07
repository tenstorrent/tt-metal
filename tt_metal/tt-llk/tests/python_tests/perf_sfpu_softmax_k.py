# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole SFPU softmax_k entry (sources/sfpu_softmax_k_perf.cpp, the body test_sfpu_softmax_k.py checks):
one call on one 4-row band of face 0 per iteration, after a datacopy into DEST. PERF_STAGE 0 runs the copy and the pack
alone, to subtract; unit: one call.
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
    SOFTMAX_K,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b


@pytest.mark.perf
@parametrize(variant=[(1, 2), (1, 8), (1, 16), (0, 8)])
def test_perf_sfpu_softmax_k(perf_report, variant):
    # parametrize hands a single axis as a one-element tuple
    if len(variant) == 1:
        (variant,) = variant
    stage, k = variant
    configuration = PerfConfig(
        "sources/sfpu_softmax_k_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
        templates=[PERF_STAGE(stage), SOFTMAX_K(softmax_k=k)],
        runtimes=[TILE_COUNT(1), LOOP_FACTOR(128)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
