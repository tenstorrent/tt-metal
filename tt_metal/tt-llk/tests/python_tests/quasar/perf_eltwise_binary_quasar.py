# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_perf_math_operations
from helpers.dest_params import (
    UnpackPath,
    dest_sync_modes,
    unpack_to_dest_modes,
)
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
)
from helpers.param_config import generate_perf_input_dimensions, parametrize
from quasar.test_eltwise_binary_quasar import (
    ELTWISE_FORMATS,
    eltwise_binary_dest_acc,
    eltwise_binary_implied_math_formats,
    eltwise_binary_math_fidelities,
)
from quasar.test_eltwise_binary_quasar import test_eltwise_binary as run_eltwise_binary
from quasar.test_eltwise_binary_quasar import (
    valid_acc_to_dest,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=ELTWISE_FORMATS,
    mathop=get_perf_math_operations,
    math_fidelity=eltwise_binary_math_fidelities,
    implied_math_format=lambda formats: eltwise_binary_implied_math_formats(
        formats, is_perf=True
    ),
    dest_acc=eltwise_binary_dest_acc,
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=lambda formats, dest_acc: unpack_to_dest_modes(
        formats, dest_acc, path=UnpackPath.FpuMath
    ),
    input_dimensions=lambda dest_acc, dest_sync: generate_perf_input_dimensions(
        dest_acc,
        dest_sync,
        use_largest_fallback=True,
    ),
    acc_to_dest=valid_acc_to_dest,
    num_faces=[4],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_binary_quasar(
    perf_report,
    formats,
    mathop,
    math_fidelity,
    implied_math_format,
    dest_acc,
    dest_sync,
    unpack_to_dest,
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
        dest_acc,
        dest_sync,
        unpack_to_dest,
        input_dimensions,
        acc_to_dest,
        num_faces,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
