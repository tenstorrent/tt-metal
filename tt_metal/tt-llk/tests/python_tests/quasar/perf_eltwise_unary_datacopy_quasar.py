# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize, runtime
from quasar.test_eltwise_unary_datacopy_quasar import (
    PERF_DATACOPY_COMBINATIONS,
    datacopy_implied_math_formats,
    datacopy_runtime_shapes,
)
from quasar.test_eltwise_unary_datacopy_quasar import (
    test_eltwise_unary_datacopy_quasar as run_eltwise_unary_datacopy,
)


@pytest.mark.nightly
@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc_data_copy_type_dest_sync=PERF_DATACOPY_COMBINATIONS,
    implied_math_format=lambda formats_dest_acc_data_copy_type_dest_sync: datacopy_implied_math_formats(
        formats_dest_acc_data_copy_type_dest_sync[0],
        is_perf=True,
    ),
    dimensions_dest_index_tile=runtime(
        lambda formats_dest_acc_data_copy_type_dest_sync: datacopy_runtime_shapes(
            formats_dest_acc_data_copy_type_dest_sync, is_perf=True
        )
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_unary_datacopy_quasar(
    perf_report,
    formats_dest_acc_data_copy_type_dest_sync,
    implied_math_format,
    dimensions_dest_index_tile,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_unary_datacopy(
        formats_dest_acc_data_copy_type_dest_sync,
        implied_math_format,
        dimensions_dest_index_tile,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
