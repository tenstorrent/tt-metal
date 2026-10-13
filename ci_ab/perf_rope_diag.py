# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CI-only diagnostic (not part of the PR): the 2x2 32-row geometry with and without the rope. Perf of the Blackhole SFPU RoPE (sources/rope_perf.cpp) in its decode form (one row per tile) and its tile form (8 to
32 rows), PERF_STAGE 0 being the datacopies alone; unit: one 32x32 DEST tile."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, PERF_STAGE, ROPE, TILE_COUNT
from test_rope import _dest_tiles, _geometry

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

VARIANTS = [
    (2, 2, 64, True, False, 0, 32, False),
    (2, 2, 64, True, False, 1, 32, False),
]
SCALE_FP32 = 0xC0000000


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_rope_diag(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    ht, wt, stride, fused, per_row, stage, tile_h, scale = variant
    geometry = _geometry(ht, wt, stride)
    if fused:
        geometry["sin_base"] = geometry["cos_base"]
    tiles = _dest_tiles(geometry)
    configuration = PerfConfig(
        "sources/rope_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
        templates=[
            ROPE(
                fused_cos_sin=fused,
                tile_h=tile_h,
                cos_sin_per_row=per_row,
                has_scale=scale,
                scale_fp32=SCALE_FP32 if scale else 0,
                **geometry,
            ),
            PERF_STAGE(stage),
        ],
        runtimes=[TILE_COUNT(tiles), LOOP_FACTOR(64)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=tiles,
            tile_count_B=1,
            tile_count_res=tiles,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
