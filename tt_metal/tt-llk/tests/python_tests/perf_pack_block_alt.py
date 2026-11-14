# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_pack_block_alt import PAIRS, TILES
from test_pack_block_alt import test_pack_block_alt as run_alt


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    pair=PAIRS,
    dest_acc=[DestAccumulation.Yes],
    block_tiles=TILES,
    l1_acc=[False, True],
    head_form=[True, False],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack_block_alt(
    perf_report,
    pair,
    dest_acc,
    block_tiles,
    l1_acc,
    head_form,
    run_types,
    loop_factor,
    is_perf,
):
    run_alt(
        pair,
        dest_acc,
        block_tiles,
        l1_acc,
        head_form,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
