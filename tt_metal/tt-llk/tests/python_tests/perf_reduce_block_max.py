# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_reduce_block_max import FORMATS, _run_reduce_block_max, block_max_num_faces

# #60571 deletes PerfRunType::L1_CONGESTION. Filter by name so this sweep stays valid
# before and after that lands, and so the kernel does not have to name the enumerator.
# respect_trigger and overlap_first_half stay at their defaults (false): the pack
# thread posts those semaphores once, before TILE_LOOP.
_PERF_RUN_TYPES = [rt for rt in ALL_PERF_RUN_TYPES if rt.name != "L1_CONGESTION"]


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    block_ct_dim=[1, 2, 4, 8],
    num_faces=block_max_num_faces,
    use_runtime=[False, True],
    run_types=[_PERF_RUN_TYPES],
    loop_factor=[32],
)
def test_perf_reduce_block_max(
    perf_report,
    formats,
    block_ct_dim,
    num_faces,
    use_runtime,
    run_types,
    loop_factor,
):
    _run_reduce_block_max(
        formats,
        block_ct_dim,
        use_runtime=use_runtime,
        num_faces=num_faces,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
