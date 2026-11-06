# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate the exact experiment configurations, including repeated bank wraparound."""

import pytest
from quasar.dest_write_experiment_cases import CASES, Kernel, Path, run_case

pytestmark = pytest.mark.quasar


@pytest.mark.parametrize(
    "path", [Path.A_HW, Path.B_DONE], ids=["unpack_dest", "srcs_dest"]
)
@pytest.mark.parametrize("name,options", CASES, ids=[name for name, _ in CASES])
def test_dest_write_experiments(name, options, path):
    run_case(name, options, path)


@pytest.mark.parametrize("depth", [1, 16])
def test_dest_write_pack1(depth):
    run_case(
        "pack1",
        {"pack1": True, "kernel": Kernel.Chain, "math_depth": depth},
        Path.B_DONE,
    )


@pytest.mark.parametrize("path", [Path.A_HW, Path.B_DONE])
@pytest.mark.parametrize("kt_dim", [1, 4, 16])
def test_dest_preload_fpu_pressure(path, kt_dim):
    from helpers.llk_params import MathFidelity
    from quasar.test_dest_preload_matmul_quasar import (
        PartialPath,
        run_dest_preload_matmul,
    )
    from quasar.test_dest_write_path_quasar import SrcsIssueOpt

    run_dest_preload_matmul(
        PartialPath(path.value),
        4,
        kt_dim,
        MathFidelity.LoFi,
        loop_factor=4,
        srcs_opt=SrcsIssueOpt.BOTH if path.srcs_writer else SrcsIssueOpt.BASE,
    )
