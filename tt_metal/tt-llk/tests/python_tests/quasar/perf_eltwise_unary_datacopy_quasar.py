# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize
from quasar.test_eltwise_unary_datacopy_quasar import (
    PERF_DATACOPY_COMBINATIONS,
)
from quasar.test_eltwise_unary_datacopy_quasar import (
    test_eltwise_unary_datacopy_quasar as run_eltwise_unary_datacopy,
)


@pytest.mark.nightly
@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc_data_copy_type_dims_dest_sync_dest_indices=PERF_DATACOPY_COMBINATIONS,
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_unary_datacopy_quasar(
    perf_report,
    formats_dest_acc_data_copy_type_dims_dest_sync_dest_indices,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_unary_datacopy(
        formats_dest_acc_data_copy_type_dims_dest_sync_dest_indices,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
