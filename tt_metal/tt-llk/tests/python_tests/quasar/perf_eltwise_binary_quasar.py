# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import PERF_RUN_TYPES_QUASAR, MathOperation
from helpers.param_config import parametrize
from quasar.test_eltwise_binary_quasar import (
    ELTWISE_FORMATS,
    eltwise_binary_dest_sync_dest_acc,
    eltwise_binary_implied_math_formats,
    eltwise_binary_input_dimensions,
    eltwise_binary_math_fidelities,
)
from quasar.test_eltwise_binary_quasar import test_eltwise_binary as run_eltwise_binary


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=ELTWISE_FORMATS,
    mathop=[MathOperation.Elwadd],
    math_fidelity=lambda mathop, formats: eltwise_binary_math_fidelities(
        mathop, formats, is_perf=True
    ),
    implied_math_format=lambda formats: eltwise_binary_implied_math_formats(
        formats, is_perf=True
    ),
    dest_sync_dest_acc=lambda formats: eltwise_binary_dest_sync_dest_acc(
        formats, is_perf=True
    ),
    input_dimensions=lambda dest_sync_dest_acc: eltwise_binary_input_dimensions(
        dest_sync_dest_acc, is_perf=True
    ),
    acc_to_dest=[False],
    num_faces=[4],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_eltwise_binary_quasar(
    perf_report,
    formats,
    mathop,
    math_fidelity,
    implied_math_format,
    dest_sync_dest_acc,
    input_dimensions,
    acc_to_dest,
    num_faces,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_binary(
        formats,
        mathop,
        math_fidelity,
        implied_math_format,
        dest_sync_dest_acc,
        input_dimensions,
        acc_to_dest,
        num_faces,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
