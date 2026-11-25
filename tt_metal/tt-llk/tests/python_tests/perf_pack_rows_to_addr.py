# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_pack_rows_to_addr import PACK_ROWS_TO_ADDR_SWEEP
from test_pack_rows_to_addr import test_pack_rows_to_addr as _test_pack_rows_to_addr


@pytest.mark.perf
@parametrize(
    **PACK_ROWS_TO_ADDR_SWEEP,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_pack_rows_to_addr(
    perf_report,
    formats,
    dest_acc,
    num_rows_to_pack,
    dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _test_pack_rows_to_addr(
        formats,
        dest_acc,
        num_rows_to_pack,
        dimensions,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
