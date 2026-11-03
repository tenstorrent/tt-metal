# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_reduce_block_max import FORMATS, _run_reduce_block_max, block_max_num_faces

PERF_LOOP_FACTOR = 32


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    block_ct_dim=[1, 2, 3, 4, 8],
    num_faces=block_max_num_faces,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_reduce_block_max(
    perf_report,
    formats,
    block_ct_dim,
    num_faces,
    run_types,
    loop_factor,
    is_perf,
):
    _run_reduce_block_max(
        formats,
        block_ct_dim,
        num_faces=num_faces,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    block_ct_dim=[1, 2, 4, 8],
    num_faces=block_max_num_faces,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_reduce_block_max_runtime(
    perf_report,
    formats,
    block_ct_dim,
    num_faces,
    run_types,
    loop_factor,
    is_perf,
):
    _run_reduce_block_max(
        formats,
        block_ct_dim,
        use_runtime=True,
        num_faces=num_faces,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
