# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_eltwise_mul_scalar_block import (
    test_eltwise_mul_scalar_block as _test_eltwise_mul_scalar_block,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]


@pytest.mark.perf
@parametrize(
    block_layout=[(1, 0), (3, 1), (4, 0)],
    dest_acc=list(DestAccumulation),
    dest_sync=list(DestSync),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_eltwise_mul_scalar_block(
    perf_report, block_layout, dest_acc, dest_sync, run_types, loop_factor, is_perf
):
    _test_eltwise_mul_scalar_block(
        block_layout,
        dest_acc,
        dest_sync,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
