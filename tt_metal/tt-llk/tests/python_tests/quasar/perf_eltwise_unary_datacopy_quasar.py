# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import flat_axes, parametrize
from quasar.test_eltwise_unary_datacopy_quasar import (
    DATACOPY_FLAT_NAMES,
    PERF_DATACOPY_COMBINATIONS,
)
from quasar.test_eltwise_unary_datacopy_quasar import (
    test_eltwise_unary_datacopy_quasar as run_eltwise_unary_datacopy,
)


@pytest.mark.nightly
@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **flat_axes(DATACOPY_FLAT_NAMES, PERF_DATACOPY_COMBINATIONS),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_unary_datacopy_quasar(
    perf_report,
    formats,
    dest_acc,
    data_copy_type,
    input_dimensions,
    dest_sync,
    dest_index,
    tile_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_unary_datacopy(
        formats,
        dest_acc,
        data_copy_type,
        input_dimensions,
        dest_sync,
        dest_index,
        tile_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
