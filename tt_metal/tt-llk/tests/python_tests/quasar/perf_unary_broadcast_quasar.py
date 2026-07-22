# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_RUN_TYPES_QUASAR, BroadcastType
from helpers.param_config import parametrize, runtime
from quasar.test_unary_broadcast_quasar import (
    INPUT_DIMENSIONS,
    UNARY_BROADCAST_FORMATS,
    get_valid_dest_acc_unary_broadcast,
)
from quasar.test_unary_broadcast_quasar import (
    test_unary_broadcast_quasar as run_unary_broadcast,
)
from quasar.test_unary_broadcast_quasar import (
    unary_broadcast_dest_sync_modes,
    unary_broadcast_implied_math_formats,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=UNARY_BROADCAST_FORMATS,
    dest_acc=get_valid_dest_acc_unary_broadcast,
    broadcast_type=[BroadcastType.Scalar],
    implied_math_format=lambda formats: unary_broadcast_implied_math_formats(
        formats, is_perf=True
    ),
    dest_sync_mode=lambda: unary_broadcast_dest_sync_modes(is_perf=True),
    input_dimensions=runtime(INPUT_DIMENSIONS),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_unary_broadcast_quasar(
    perf_report,
    formats,
    dest_acc,
    broadcast_type,
    implied_math_format,
    dest_sync_mode,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_unary_broadcast(
        formats,
        dest_acc,
        broadcast_type,
        implied_math_format,
        dest_sync_mode,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
