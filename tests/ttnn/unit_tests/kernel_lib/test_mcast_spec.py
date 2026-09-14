# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ProgramSpec attachment and device argument frontend contracts."""
from pathlib import Path
import subprocess
import pytest


@pytest.mark.parametrize("case", ["Smoke", "Matrix"])
def test_mcast_spec_device(case):
    repo = Path(__file__).resolve().parents[4]
    subprocess.run(
        [str(repo / "build/test/ttnn/unit_tests_ttnn"), f"--gtest_filter=McastHostFixture.SpecDevice{case}"],
        check=True,
    )
