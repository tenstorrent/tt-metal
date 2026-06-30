# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from fast_untilize_common import (
    FAST_UNTILIZE_CT_DIMS,
    FAST_UNTILIZE_RT_DIMS,
    FAST_UNTILIZE_TILE_C,
    FAST_UNTILIZE_TILE_R,
    fast_untilize_32b_dest_modes,
    fast_untilize_formats,
)
from helpers.llk_params import Fp32DestMode, PerfRunType
from helpers.param_config import parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    TILE_COUNT,
    generate_input_dim,
)


def baseline_pack_untilize_block_ct_dim(ct_dim, is_32b_dest_en):
    max_dest_tiles = 4 if is_32b_dest_en == Fp32DestMode.Yes else 8
    for candidate in range(min(ct_dim, max_dest_tiles), 0, -1):
        if ct_dim % candidate == 0:
            return candidate
    return 1


@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=fast_untilize_formats(),
    is_32b_dest_en=fast_untilize_32b_dest_modes,
    rt_dim=FAST_UNTILIZE_RT_DIMS,
    ct_dim=FAST_UNTILIZE_CT_DIMS,
    loop_factor=[1, 4, 16],
)
def test_perf_fast_untilize_baseline_compare(
    perf_report, formats, is_32b_dest_en, rt_dim, ct_dim, loop_factor
):
    tile_count = rt_dim * ct_dim
    dimensions = (rt_dim * FAST_UNTILIZE_TILE_R, ct_dim * FAST_UNTILIZE_TILE_C)
    block_ct_dim = baseline_pack_untilize_block_ct_dim(ct_dim, is_32b_dest_en)

    configuration = PerfConfig(
        "sources/pack_untilize_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[generate_input_dim(dimensions, dimensions, block_ct_dim)],
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(loop_factor),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        compile_time_formats=True,
        is_32b_dest_en=is_32b_dest_en,
    )

    configuration.run(perf_report)
