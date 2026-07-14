# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize, runtime
from quasar.test_pack_l1_acc_quasar import (
    ALL_PACK_L1_ACC_COMBINATIONS,
    PERF_ONLY_INPUT_DIMENSIONS,
    pack_l1_acc_dest_sync_modes,
    pack_l1_acc_implied_math_formats,
)
from quasar.test_pack_l1_acc_quasar import test_pack_l1_acc_quasar as run_pack_l1_acc

@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc=ALL_PACK_L1_ACC_COMBINATIONS,
    implied_math_format=lambda formats_dest_acc: pack_l1_acc_implied_math_formats(
        formats_dest_acc, is_perf=True
    ),
    dest_sync_mode=lambda: pack_l1_acc_dest_sync_modes(is_perf=True),
    input_dimensions=runtime(PERF_ONLY_INPUT_DIMENSIONS),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack_l1_acc_quasar(
    perf_report,
    formats_dest_acc,
    implied_math_format,
    dest_sync_mode,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_pack_l1_acc(
        formats_dest_acc,
        implied_math_format,
        dest_sync_mode,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
