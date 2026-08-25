# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import DestSync
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_pack import PERF_PACK_COMBINATIONS
from test_pack import test_pack as run_pack


@pytest.mark.perf
@parametrize(
    formats_dest_acc_dims_relu=PERF_PACK_COMBINATIONS,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack(
    perf_report,
    formats_dest_acc_dims_relu,
    run_types,
    loop_factor,
    is_perf,
):
    formats, dest_acc, input_dimensions, relu_type = formats_dest_acc_dims_relu
    run_pack(
        formats,
        dest_acc,
        input_dimensions,
        relu_type,
        dest_sync=DestSync.Half,
        dest_index=0,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
