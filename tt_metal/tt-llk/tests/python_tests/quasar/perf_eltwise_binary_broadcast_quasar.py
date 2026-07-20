# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.llk_params import BroadcastType, MathOperation, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize
from quasar.test_eltwise_binary_broadcast_quasar import (
    BINARY_BROADCAST_FORMATS,
    binary_broadcast_dest_sync_modes,
    binary_broadcast_implied_math_formats,
    binary_broadcast_input_dimensions,
    binary_broadcast_math_fidelities,
)
from quasar.test_eltwise_binary_broadcast_quasar import (
    test_eltwise_binary_broadcast_quasar as run_eltwise_binary_broadcast,
)

@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=BINARY_BROADCAST_FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    mathop=[MathOperation.Elwadd],
    broadcast_type=[BroadcastType.Scalar],
    math_fidelity=lambda formats, mathop: binary_broadcast_math_fidelities(
        formats, mathop, is_perf=True
    ),
    implied_math_format=lambda formats: binary_broadcast_implied_math_formats(
        formats, is_perf=True
    ),
    dest_sync_mode=lambda: binary_broadcast_dest_sync_modes(is_perf=True),
    input_dimensions=lambda dest_acc, dest_sync_mode: binary_broadcast_input_dimensions(
        dest_acc, dest_sync_mode, is_perf=True
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
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
    dest_sync_mode,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_binary_broadcast(
        formats,
        dest_acc,
        mathop,
        broadcast_type,
        math_fidelity,
        implied_math_format,
        dest_sync_mode,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
