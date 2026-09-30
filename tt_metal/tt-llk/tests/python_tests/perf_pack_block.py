# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_pack import PACK_BLOCK_SWEEP
from test_pack import test_pack as run_pack


# The block pack (one pack program run per dest block) on the pack_test.cpp kernel: the same rows as perf_pack.py
# without the relu axis and with blocks of up to 16 tiles, so the cycles per tile can be read against perf_pack.py's
# per-tile pack of the same formats and dest modes.
@pytest.mark.perf
@parametrize(
    **{**PACK_BLOCK_SWEEP, "dest_index": [0]},
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack_block(
    perf_report,
    formats,
    dest_acc,
    input_dimensions,
    relu_type,
    dest_sync,
    dest_index,
    run_types,
    loop_factor,
    is_perf,
):
    run_pack(
        formats,
        dest_acc,
        input_dimensions,
        relu_type,
        dest_sync,
        dest_index,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
        pack_block=True,
    )
