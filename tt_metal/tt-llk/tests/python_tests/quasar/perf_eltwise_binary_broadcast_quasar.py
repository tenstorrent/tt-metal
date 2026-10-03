# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import (
    get_perf_math_operations,
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
)
from helpers.param_config import parametrize
from quasar.test_eltwise_binary_broadcast_quasar import (
    BINARY_BROADCAST_PERF_FORMATS,
    BROADCAST_TYPES,
    binary_broadcast_acc_to_dest_modes,
    binary_broadcast_dest_sync_modes,
    binary_broadcast_implied_math_formats,
    binary_broadcast_input_dimensions,
    binary_broadcast_tile_dimensions,
    skip_if_quasar_binary_broadcast_unsupported,
)
from quasar.test_eltwise_binary_broadcast_quasar import (
    test_eltwise_binary_broadcast_quasar as run_eltwise_binary_broadcast,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=BINARY_BROADCAST_PERF_FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    mathop=get_perf_math_operations,
    broadcast_type=BROADCAST_TYPES,
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(formats, mathop),
    implied_math_format=lambda formats: binary_broadcast_implied_math_formats(
        formats, is_perf=True
    ),
    dest_sync=lambda: binary_broadcast_dest_sync_modes(is_perf=True),
    unpack_to_dest=[False],
    tile_dimensions=lambda formats, broadcast_type: binary_broadcast_tile_dimensions(
        formats, broadcast_type, is_perf=True
    ),
    input_dimensions=lambda dest_acc, dest_sync, tile_dimensions: binary_broadcast_input_dimensions(
        dest_acc, dest_sync, tile_dimensions, is_perf=True
    ),
    acc_to_dest=lambda input_dimensions, tile_dimensions: binary_broadcast_acc_to_dest_modes(
        input_dimensions, tile_dimensions, is_perf=True
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_binary_broadcast_quasar(
    perf_report,
    formats,
    dest_acc,
    mathop,
    broadcast_type,
    math_fidelity,
    implied_math_format,
    dest_sync,
    unpack_to_dest,
    tile_dimensions,
    input_dimensions,
    acc_to_dest,
    run_types,
    loop_factor,
    is_perf,
):
    skip_if_quasar_binary_broadcast_unsupported(
        tile_dimensions, math_fidelity, acc_to_dest
    )
    run_eltwise_binary_broadcast(
        formats,
        dest_acc,
        mathop,
        broadcast_type,
        math_fidelity,
        implied_math_format,
        dest_sync,
        unpack_to_dest,
        tile_dimensions,
        input_dimensions,
        acc_to_dest,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
