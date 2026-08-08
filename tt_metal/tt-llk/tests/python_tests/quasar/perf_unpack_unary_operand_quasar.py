# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize, runtime
from quasar.test_unpack_unary_operand_quasar import (
    UNPACK_FORMATS,
)
from quasar.test_unpack_unary_operand_quasar import (
    test_unpack_unary_operand_quasar as run_unpack_unary_operand,
)
from quasar.test_unpack_unary_operand_quasar import (
    unpack_unary_dest_acc_modes,
    unpack_unary_dest_sync_modes,
    unpack_unary_engines,
    unpack_unary_formats,
    unpack_unary_input_dimensions,
    unpack_unary_tile_dimensions,
    unpack_unary_transpose_modes,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=lambda: unpack_unary_formats(UNPACK_FORMATS, is_perf=True),
    dest_acc=lambda formats: unpack_unary_dest_acc_modes(formats, is_perf=True),
    dest_sync_mode=lambda: unpack_unary_dest_sync_modes(is_perf=True),
    transpose=unpack_unary_transpose_modes,
    unpacker_sel=unpack_unary_engines,
    tile_dimensions=runtime(
        lambda formats, transpose, unpacker_sel: unpack_unary_tile_dimensions(
            formats, transpose, unpacker_sel, is_perf=True
        )
    ),
    input_dimensions=runtime(
        lambda dest_acc, dest_sync_mode, tile_dimensions: unpack_unary_input_dimensions(
            dest_acc, dest_sync_mode, tile_dimensions, is_perf=True
        )
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_unpack_unary_operand_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync_mode,
    transpose,
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
        dest_sync_mode,
        transpose,
        unpacker_sel,
        input_dimensions,
        tile_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
