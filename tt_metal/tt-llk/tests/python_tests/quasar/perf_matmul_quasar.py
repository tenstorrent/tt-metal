# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_RUN_TYPES_QUASAR, Transpose
from helpers.param_config import parametrize
from quasar.test_matmul_quasar import (
    MATMUL_FORMAT,
    matmul_dest_acc_modes,
    matmul_dest_sync_modes,
    matmul_dimensions,
    matmul_enable_direct_indexing,
    matmul_implied_math_formats,
    matmul_math_fidelities,
    matmul_register_format_hints,
)
from quasar.test_matmul_quasar import test_matmul as run_matmul


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    format=MATMUL_FORMAT,
    math_fidelity=matmul_math_fidelities,
    dest_sync_mode=lambda: matmul_dest_sync_modes(is_perf=True),
    dest_acc=matmul_dest_acc_modes,
    dimensions=lambda dest_acc, dest_sync_mode: matmul_dimensions(
        dest_acc,
        dest_sync_mode,
        exact_dest_fill=True,
    ),
    implied_math_format=lambda format: matmul_implied_math_formats(
        format, is_perf=True
    ),
    register_format_hint=matmul_register_format_hints,
    enable_direct_indexing=matmul_enable_direct_indexing,
    transpose=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_matmul_quasar(
    perf_report,
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    dimensions,
    format,
    implied_math_format,
    register_format_hint,
    enable_direct_indexing,
    transpose,
    run_types,
    loop_factor,
    is_perf,
):
    run_matmul(
        math_fidelity,
        dest_sync_mode,
        dest_acc,
        dimensions,
        format,
        implied_math_format,
        register_format_hint,
        enable_direct_indexing,
        transpose,
        run_types,
        loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
