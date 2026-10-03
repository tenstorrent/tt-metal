# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the csa_index_remap SFPU op (sources/csa_index_remap_perf.cpp); unit: one section of three tiles,
PERF_STAGE 0 being the datacopies alone."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    PERF_STAGE,
    TILE_COUNT,
)
from test_csa_index_remap import CSA_REMAP

pytestmark = [skip_for_wormhole, skip_for_quasar]

# (row offset, stage)
VARIANTS = [(0, 1), (256, 1), (0, 0)]


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_csa_index_remap(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    row_offset, stage = variant
    configuration = PerfConfig(
        "sources/csa_index_remap_perf.cpp",
        InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32),
        run_types=[PerfRunType.L1_TO_L1],
        templates=[CSA_REMAP(row_offset), DEST_SYNC(DestSync.Half), PERF_STAGE(stage)],
        runtimes=[DEST_INDEX(0), TILE_COUNT(3), LOOP_FACTOR(64)],
        variant_stimuli=StimuliConfig(
            None,
            DataFormat.UInt32,
            None,
            DataFormat.UInt32,
            DataFormat.UInt32,
            tile_count_A=9,
            tile_count_B=1,
            tile_count_res=9,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )
    configuration.run(perf_report)
