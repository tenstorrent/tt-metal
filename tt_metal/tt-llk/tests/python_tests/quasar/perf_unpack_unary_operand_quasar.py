# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import flat_axes, parametrize
from quasar.test_unpack_unary_operand_quasar import (
    PERF_UNPACK_UNARY_OPERAND_COMBINATIONS,
    UNPACK_UNARY_OPERAND_FLAT_NAMES,
)
from quasar.test_unpack_unary_operand_quasar import (
    test_unpack_unary_operand_quasar as run_unpack_unary_operand,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **flat_axes(
        UNPACK_UNARY_OPERAND_FLAT_NAMES, PERF_UNPACK_UNARY_OPERAND_COMBINATIONS
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_unpack_unary_operand_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    transpose_en,
    unpacker_sel,
    input_dimensions,
    tile_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_unpack_unary_operand(
        formats,
        dest_acc,
        dest_sync,
        transpose_en,
        unpacker_sel,
        input_dimensions,
        tile_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
