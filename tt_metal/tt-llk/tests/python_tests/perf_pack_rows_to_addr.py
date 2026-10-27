# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.param_config import parametrize
from quasar.test_pack_rows_to_addr_quasar import (
    PACK_ROWS_PERF_SWEEP,
    run_pack_rows_to_addr_perf,
)


@pytest.mark.perf
@blackhole_only
@parametrize(**PACK_ROWS_PERF_SWEEP)
def test_perf_pack_rows_to_addr(perf_report, formats, num_rows_to_pack):
    run_pack_rows_to_addr_perf(perf_report, formats, num_rows_to_pack)
