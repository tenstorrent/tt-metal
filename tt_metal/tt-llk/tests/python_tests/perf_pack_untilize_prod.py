# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""perf_pack_untilize's standard sweep with production's packer input (data_format.cpp): a block float input from a 16-bit
DEST packs from the output format family. perf_pack_untilize keeps main's packer input, so the perf gate compares like with like.
"""
import pytest
from helpers.llk_params import DestSync, PackUntilizeBenchmark
from helpers.param_config import parametrize
from perf_pack_untilize import (
    PACK_UNTILIZE_RUN_TYPES,
    pack_untilize_block_ct_dim,
    pack_untilize_ct_dims,
    pack_untilize_dest_acc_modes,
    pack_untilize_formats,
    pack_untilize_rt_dims,
)
from test_pack_untilize import test_pack_untilize as run_pack_untilize

pytestmark = pytest.mark.llk_sweep


@pytest.mark.perf
@parametrize(
    benchmark=[PackUntilizeBenchmark.Standard],
    formats=pack_untilize_formats,
    dest_acc=pack_untilize_dest_acc_modes,
    full_rt_dim=pack_untilize_rt_dims,
    full_ct_dim=pack_untilize_ct_dims,
    dest_sync=[DestSync.Half],
    run_types=[PACK_UNTILIZE_RUN_TYPES],
    loop_factor=[64],
)
def test_perf_pack_untilize_prod(
    perf_report,
    benchmark,
    formats,
    dest_acc,
    full_rt_dim,
    full_ct_dim,
    dest_sync,
    run_types,
    loop_factor,
):
    run_pack_untilize(
        formats,
        dest_acc,
        [full_rt_dim * 32, full_ct_dim * 32],
        dest_sync,
        tile_dst_ct_offset=0,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        block_ct_dim=pack_untilize_block_ct_dim(
            benchmark, formats, dest_acc, full_ct_dim
        ),
        compile_time_formats=False,
        benchmark=benchmark,
        production_pack_src=True,
    )
