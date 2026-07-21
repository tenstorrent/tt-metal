# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import EltwiseBinaryReuseDestType, PERF_RUN_TYPES_QUASAR
from helpers.param_config import parametrize, runtime
from quasar.test_eltwise_binary_reuse_dest_quasar import (
    INPUT_DIMENSIONS,
    OUTPUT_DIMENSIONS,
    REUSE_DEST_FORMATS,
    reuse_dest_dest_sync_modes,
    reuse_dest_math_fidelities,
    reuse_dest_mathops,
)
from quasar.test_eltwise_binary_reuse_dest_quasar import (
    test_eltwise_binary_reuse_dest_quasar as run_eltwise_binary_reuse_dest,
)

@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=REUSE_DEST_FORMATS,
    mathop=lambda formats: reuse_dest_mathops(formats, is_perf=True),
    math_fidelity=lambda mathop: reuse_dest_math_fidelities(mathop, is_perf=True),
    reuse_dest_type=[
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ],
    dest_sync_mode=lambda: reuse_dest_dest_sync_modes(is_perf=True),
    input_dimensions=runtime(INPUT_DIMENSIONS),
    output_dimensions=runtime(OUTPUT_DIMENSIONS),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_eltwise_binary_reuse_dest_quasar(
    perf_report,
    formats,
    mathop,
    reuse_dest_type,
    math_fidelity,
    dest_sync_mode,
    input_dimensions,
    output_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_binary_reuse_dest(
        formats,
        mathop,
        reuse_dest_type,
        math_fidelity,
        dest_sync_mode,
        input_dimensions,
        output_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
