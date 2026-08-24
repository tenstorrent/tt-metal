# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import blackhole_only
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import ROPE, TILE_COUNT
from test_rope import (
    DENSE_STRIDE,
    FORMATS,
    TILE_SLOT_STRIDE,
    _dest_tiles,
    _geometry,
    _heads,
    _stimuli,
)

pytestmark = blackhole_only


@pytest.mark.perf
@parametrize(stride=[TILE_SLOT_STRIDE, DENSE_STRIDE], wt=[1, 2])
def test_perf_rope(perf_report, stride, wt):
    ht = _heads(stride, wt)[-1]
    geometry = _geometry(ht, wt, stride)
    tiles = _dest_tiles(geometry)
    dest = _stimuli(geometry, tiles, seed=101)

    configuration = PerfConfig(
        "sources/rope_test.cpp",
        FORMATS,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            ROPE(
                has_scale=False,
                scale_fp32=0,
                **geometry,
            ),
        ],
        runtimes=[TILE_COUNT(tiles)],
        variant_stimuli=StimuliConfig(
            dest.flatten(),
            FORMATS.input_format,
            torch.zeros(ELEMENTS_PER_TILE, dtype=torch.bfloat16),
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=tiles,
            tile_count_B=1,
            tile_count_res=tiles,
        ),
    )
    configuration.run(perf_report)
