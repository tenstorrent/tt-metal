# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_topk_pack_supplied_indices import (
    TOPK_PACK_SUPPLIED_INDICES_SWEEP,
)
from test_topk_pack_supplied_indices import (
    test_topk_pack_supplied_indices as _test_topk_pack_supplied_indices,
)

pytestmark = [skip_for_quasar, skip_for_wormhole]


@pytest.mark.perf
@parametrize(
    **TOPK_PACK_SUPPLIED_INDICES_SWEEP,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_topk_pack_supplied_indices(
    perf_report, case, alias_sync, run_types, loop_factor, is_perf
):
    _test_topk_pack_supplied_indices(
        case,
        alias_sync,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
