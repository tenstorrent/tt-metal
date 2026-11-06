# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Table test of the Wormhole isolate mocks' MOP stream planning (perf_mock_plan.h), compiled and run on the host.

The plan decides which emitter runs a stand-in stream and how many MOPs, loop counts and replay handshakes it takes. It is
free of hardware headers, so the periods 16/17/18, run counts 127/128, a_count 31/32, odd reuse and zero work cases run
without a device.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_INCLUDE = _HERE.parents[1] / "helpers" / "include"
_TABLE = _HERE / "perf_mock_plan_table.cpp"


def test_perf_mock_plan_table(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("no host C++ compiler")
    binary = tmp_path / "perf_mock_plan_table"
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{_INCLUDE}",
            str(_TABLE),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout
