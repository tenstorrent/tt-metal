# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_rmsnorm_clear_product_tile import (
    test_rmsnorm_clear_product_tile as _test_rmsnorm_clear_product_tile,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half, DestSync.Full],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_rmsnorm_clear_product_tile(
    perf_report, dest_acc, dest_sync, run_types, loop_factor, is_perf
):
    _test_rmsnorm_clear_product_tile(
        dest_acc,
        dest_sync,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
