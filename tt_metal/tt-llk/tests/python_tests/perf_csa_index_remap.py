# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.llk_params import DestSync
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_csa_index_remap import test_csa_index_remap as _test_csa_index_remap

pytestmark = [skip_for_wormhole, skip_for_quasar]


@pytest.mark.perf
@parametrize(
    row_offset=[0, 1, 7, 8, 255, 256, 32768],
    dst_index=[0, 1],
    dest_sync=list(DestSync),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_csa_index_remap(
    perf_report, row_offset, dst_index, dest_sync, run_types, loop_factor, is_perf
):
    _test_csa_index_remap(
        row_offset,
        dst_index,
        dest_sync,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
