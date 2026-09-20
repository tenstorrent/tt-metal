# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-attached compiler checks and native host/spec integration launchers.

Run this launcher before tests that open a Python device. Compiler checks run
in child processes too: closing a Python device does not release its process-wide
UMD ownership, which would block a later native child.
"""

from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize(
    "violation,with_dense_group",
    [
        ("chain-receive", False),
        ("chain-receive", True),
        ("wrong-noc", False),
        ("wrong-noc", True),
        ("source-unused", False),
        ("source-data-ready", False),
        ("source-consumer-ready", False),
    ],
)
def test_forwarding_receive_compile_contract(with_dense_group, violation):
    compiler_checks = Path(__file__).with_name("mcast_compile_contracts.py")
    case = (
        f"{compiler_checks}::test_forwarding_receive_compile_contract"
        f"[violation={violation}-with_dense_group={with_dense_group}]"
    )
    subprocess.run([sys.executable, "-m", "pytest", case, "-q", "-x"], check=True)


def test_host_cpp_contract():
    repo = Path(__file__).resolve().parents[4]
    subprocess.run(
        [str(repo / "build/test/ttnn/unit_tests_ttnn"), "--gtest_filter=Mcast*-McastHostFixture.SpecDevice*"],
        check=True,
    )


@pytest.mark.parametrize("case", ["Smoke", "Matrix"])
def test_mcast_spec_device(case):
    repo = Path(__file__).resolve().parents[4]
    subprocess.run(
        [str(repo / "build/test/ttnn/unit_tests_ttnn"), f"--gtest_filter=McastHostFixture.SpecDevice{case}"],
        check=True,
    )
