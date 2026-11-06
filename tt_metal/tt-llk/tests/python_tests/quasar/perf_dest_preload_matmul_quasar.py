# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""FPU pressure at three K depths using the unchanged existing matmul kernel."""

import pytest
from helpers.llk_params import MathFidelity
from quasar.test_dest_preload_matmul_quasar import PartialPath, run_dest_preload_matmul
from quasar.test_dest_write_path_quasar import SrcsIssueOpt

pytestmark = [pytest.mark.perf, pytest.mark.quasar]


@pytest.mark.parametrize("path", [PartialPath.A_HW, PartialPath.B_DONE])
@pytest.mark.parametrize("kt_dim", [1, 4, 16])
def test_perf_dest_preload_matmul_quasar(perf_report, path, kt_dim):
    run_dest_preload_matmul(
        path,
        4,
        kt_dim,
        MathFidelity.LoFi,
        loop_factor=4,
        srcs_opt=SrcsIssueOpt.BOTH if path.srcs_writer else SrcsIssueOpt.BASE,
        is_perf=True,
        perf_report=perf_report,
    )
