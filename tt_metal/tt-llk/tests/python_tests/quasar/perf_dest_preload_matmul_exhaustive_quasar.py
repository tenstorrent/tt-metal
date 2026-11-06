# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import MathFidelity
from helpers.param_config import parametrize
from quasar.test_dest_preload_matmul_quasar import (
    MATMUL_CT_SWEEP,
    MATMUL_FIDELITIES,
    MATMUL_KT_SWEEP,
    MATMUL_SRCS_OPTS,
    PartialPath,
    run_dest_preload_matmul,
)
from quasar.test_dest_write_path_quasar import (
    PERF_LOOP_FACTOR,
    PERF_RUN_TYPES,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    path=list(PartialPath),
    ct_dim=MATMUL_CT_SWEEP,
    kt_dim=MATMUL_KT_SWEEP,
    math_fidelity=MATMUL_FIDELITIES,
    run_types=PERF_RUN_TYPES,
)
def test_perf_dest_preload_matmul_quasar(
    perf_report, path, ct_dim, kt_dim, math_fidelity, run_types
):
    run_dest_preload_matmul(
        path,
        ct_dim,
        kt_dim,
        math_fidelity,
        run_types=run_types,
        loop_factor=PERF_LOOP_FACTOR,
        is_perf=True,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    path=[PartialPath.B_DRAIN, PartialPath.B_DONE],
    ct_dim=MATMUL_CT_SWEEP,
    kt_dim=MATMUL_KT_SWEEP,
    srcs_opt=MATMUL_SRCS_OPTS,
    run_types=PERF_RUN_TYPES,
)
def test_perf_dest_preload_matmul_srcs_opt_quasar(
    perf_report, path, ct_dim, kt_dim, srcs_opt, run_types
):
    run_dest_preload_matmul(
        path,
        ct_dim,
        kt_dim,
        MathFidelity.LoFi,
        run_types=run_types,
        loop_factor=PERF_LOOP_FACTOR,
        is_perf=True,
        perf_report=perf_report,
        srcs_opt=srcs_opt,
    )
