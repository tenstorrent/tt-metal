# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run native multicast gtests in a fresh pytest process.

Explicitly invoke this file through scripts/run_safe_pytest.sh, separately from Python
device tests. The runtime retains the chip driver until process exit, even after closing
a device. This filename intentionally avoids default pytest collection so a native child
cannot wait on a driver held by earlier Python tests in the same session.
"""

from pathlib import Path
import subprocess
import pytest


def test_host_cpp_contract():
    repo = Path(__file__).resolve().parents[4]
    subprocess.run(
        [
            str(repo / "build/test/ttnn/unit_tests_ttnn"),
            "--gtest_filter=Mcast*:GroupNormMcastGeometry.*-McastHostFixture.SpecDevice*",
        ],
        check=True,
    )


@pytest.mark.parametrize("case", ["Smoke", "Matrix"])
def test_mcast_spec_device(case):
    repo = Path(__file__).resolve().parents[4]
    subprocess.run(
        [str(repo / "build/test/ttnn/unit_tests_ttnn"), f"--gtest_filter=McastHostFixture.SpecDevice{case}"],
        check=True,
    )
