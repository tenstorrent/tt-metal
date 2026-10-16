# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# FPU matmul vs TRISC3 SFPU dequant cost per quantization group. K is fixed at 256 and
# the group count varies (group 256/128/64 -> 1/2/4 groups), so per-group SFPU cost and
# the one-off s_a epilogue separate as slope and intercept.

import pytest
from helpers.chip_architecture import is_4row_arch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR_4_TRISC,
    DestSync,
)
from helpers.param_config import parametrize
from quasar.test_matmul_int4_dequant_quasar import (
    PERF_GROUP_DIMS,
)
from quasar.test_matmul_int4_dequant_quasar import (
    test_matmul_int4_dequant_quasar as run_matmul_int4_dequant,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    weight_format=[DataFormat.Int4],
    dest_sync_mode=[DestSync.Half],
    matmul_dims=PERF_GROUP_DIMS,
    # Int8_2x (the fast int8 matmul) exists only on the 4-row variant.
    register_format_hint=[None, DataFormat.Int8_2x] if is_4row_arch() else [None],
    run_types=PERF_RUN_TYPES_QUASAR_4_TRISC,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
)
def test_perf_matmul_int4_dequant_quasar(
    perf_report,
    weight_format,
    dest_sync_mode,
    matmul_dims,
    register_format_hint,
    run_types,
    loop_factor,
):
    run_matmul_int4_dequant(
        weight_format,
        dest_sync_mode,
        matmul_dims,
        register_format_hint,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=True,
        perf_report=perf_report,
    )
