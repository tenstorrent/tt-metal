# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import DestSync, PerfRunType
from helpers.param_config import parametrize
from test_pack_untilize import (
    pack_untilize_block_ct_dim,
    pack_untilize_cases,
)
from test_pack_untilize import test_pack_untilize as run_pack_untilize

PACK_UNTILIZE_RUN_TYPES = [
    PerfRunType.L1_TO_L1,
    PerfRunType.UNPACK_ISOLATE,
    PerfRunType.MATH_ISOLATE,
    PerfRunType.PACK_ISOLATE,
]


@pytest.mark.perf
@parametrize(
    case=pack_untilize_cases,
    dest_sync=[DestSync.Half],
    tile_dst_ct_offset=[0],
    run_types=[PACK_UNTILIZE_RUN_TYPES],
    loop_factor=[64],
    is_perf=[True],
)
def test_perf_pack_untilize(
    perf_report,
    case,
    dest_sync,
    tile_dst_ct_offset,
    run_types,
    loop_factor,
    is_perf,
):
    block_ct_dim = pack_untilize_block_ct_dim(case)

    run_pack_untilize(
        case,
        dest_sync,
        tile_dst_ct_offset,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        block_ct_dim=block_ct_dim,
    )
