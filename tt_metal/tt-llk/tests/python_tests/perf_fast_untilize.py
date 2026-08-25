# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from fast_untilize_common import (
    FAST_UNTILIZE_DEST_SYNC_MODES,
    FAST_UNTILIZE_TILE_C,
    FAST_UNTILIZE_TILE_R,
    fast_untilize_dest_acc_modes,
    fast_untilize_formats,
)
from helpers.llk_params import DestSync, PerfRunType
from helpers.param_config import generate_perf_input_dimensions, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    TILE_COUNT,
    generate_input_dim,
)


def _fast_untilize_rt_ct(dest_acc, dest_sync):
    """Dest-fill shapes with ct>=2. Tall dest-fill is (max, 1); remap to (max//2, 2)."""
    tiles = []
    seen = set()
    for dims in generate_perf_input_dimensions(dest_acc, dest_sync):
        rt_dim = dims[0] // FAST_UNTILIZE_TILE_R
        ct_dim = dims[1] // FAST_UNTILIZE_TILE_C
        # Kernel static_assert(FULL_CT_DIM >= 2); ct=1 uses the standard fallback.
        if ct_dim < 2:
            rt_dim = max(rt_dim // 2, 1)
            ct_dim = 2
        pair = (rt_dim, ct_dim)
        if pair not in seen:
            seen.add(pair)
            tiles.append(pair)
    # Remainder blocking: one non-power-of-two width on Half dest.
    if dest_sync == DestSync.Half and (1, 3) not in seen:
        tiles.append((1, 3))
    return tiles


@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=fast_untilize_formats(),
    dest_acc=fast_untilize_dest_acc_modes,
    dest_sync=FAST_UNTILIZE_DEST_SYNC_MODES,
    rt_ct=lambda dest_acc, dest_sync: _fast_untilize_rt_ct(dest_acc, dest_sync),
    loop_factor=[32],
)
def test_perf_fast_untilize(
    perf_report, formats, dest_acc, dest_sync, rt_ct, loop_factor
):
    rt_dim, ct_dim = rt_ct
    tile_count = rt_dim * ct_dim
    dimensions = (rt_dim * FAST_UNTILIZE_TILE_R, ct_dim * FAST_UNTILIZE_TILE_C)

    configuration = PerfConfig(
        "sources/fast_untilize_test.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[generate_input_dim(dimensions, dimensions), DEST_SYNC(dest_sync)],
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
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
