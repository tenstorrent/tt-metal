# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the dest-reuse op in its callers' order (sources/rmsnorm_bcast_scalar_dest_reuse_sequence_perf.cpp): per
iteration a LoFi ELWSUB and tile_cnt - 1 HiFi ELWMULs of one tile each, every call with its init; unit: one tile.
"""

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
    RMSNORM_DEST_REUSE,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

# (fidelity of the multiplies, tiles per iteration, whole_tile, the callers' SFPU steps)
VARIANTS = [
    (fidelity, tile_cnt, whole_tile, sfpu)
    for fidelity in (MathFidelity.HiFi2, MathFidelity.HiFi4)
    for tile_cnt in (2, 3, 4)
    for whole_tile in (False, True)
    for sfpu in (False, True)
]


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_rmsnorm_bcast_scalar_dest_reuse_sequence(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    fidelity, tile_cnt, whole_tile, sfpu = variant
    configuration = PerfConfig(
        "sources/rmsnorm_bcast_scalar_dest_reuse_sequence_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            MATH_FIDELITY(fidelity),
            RMSNORM_DEST_REUSE(rmsnorm_whole_tile=whole_tile, rmsnorm_shadow_sfpu=sfpu),
        ],
        runtimes=[TILE_COUNT(tile_cnt), LOOP_FACTOR(64)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=tile_cnt,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
