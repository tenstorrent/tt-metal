# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small decision suite. Historical Cartesian sweep: perf_dest_write_path_exhaustive_quasar.py."""

import pytest
from quasar.dest_write_experiment_cases import CASES, Kernel, Path, run_case

pytestmark = [pytest.mark.perf, pytest.mark.quasar]


@pytest.mark.parametrize(
    "path", [Path.A_HW, Path.B_DONE], ids=["unpack_dest", "srcs_dest"]
)
@pytest.mark.parametrize("name,options", CASES, ids=[name for name, _ in CASES])
def test_perf_dest_write_experiments(perf_report, name, options, path):
    run_case(name, options, path, is_perf=True, perf_report=perf_report)


@pytest.mark.parametrize("depth", [1, 16])
def test_perf_dest_write_pack1(perf_report, depth):
    run_case(
        "pack1",
        {"pack1": True, "kernel": Kernel.Chain, "math_depth": depth},
        Path.B_DONE,
        is_perf=True,
        perf_report=perf_report,
    )
