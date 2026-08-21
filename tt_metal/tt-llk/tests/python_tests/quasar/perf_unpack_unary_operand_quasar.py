# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize, runtime
from quasar.test_unpack_unary_operand_quasar import (
    PERF_UNPACK_UNARY_OPERAND_COMBINATIONS,
)
from quasar.test_unpack_unary_operand_quasar import (
    test_unpack_unary_operand_quasar as run_unpack_unary_operand,
)
from quasar.test_unpack_unary_operand_quasar import (
    unpack_unary_operand_runtime_shapes,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_transpose_unpack_sel=PERF_UNPACK_UNARY_OPERAND_COMBINATIONS,
    dimensions_and_tile=runtime(
        lambda formats_dest_acc_sync_transpose_unpack_sel: unpack_unary_operand_runtime_shapes(
            formats_dest_acc_sync_transpose_unpack_sel, is_perf=True
        )
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_unpack_unary_operand_quasar(
    perf_report,
    formats_dest_acc_sync_transpose_unpack_sel,
    dimensions_and_tile,
    run_types,
    loop_factor,
    is_perf,
):
    run_unpack_unary_operand(
        formats_dest_acc_sync_transpose_unpack_sel,
        dimensions_and_tile,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
