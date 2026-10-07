# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the actual cleanup trap with errexit, without launching MPI or devices."""

from pathlib import Path
import re
import subprocess

import pytest


@pytest.mark.parametrize("rankfile", ["empty", "present", "absent"])
@pytest.mark.parametrize("run_status", [0, 7])
def test_cleanup_preserves_run_status(tmp_path, rankfile, run_status):
    script = Path(__file__).resolve().parents[1] / "run_fabric_tests.sh"
    match = re.search(r"^cleanup_run_artifacts\(\) \{\n.*?^\}\n", script.read_text(), re.MULTILINE | re.DOTALL)
    assert match is not None
    path = tmp_path / "rankfile"
    if rankfile == "present":
        path.write_text("rank 0=localhost slot=0\n")
    # Arguments are passed separately, never interpolated into shell code.
    isolated = "Z_RANKFILE=$1\n" + match.group(0) + '\ntrap cleanup_run_artifacts EXIT\nexit "$2"\n'
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-o",
            "pipefail",
            "-c",
            isolated,
            "cleanup-test",
            "" if rankfile == "empty" else str(path),
            str(run_status),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == run_status, result.stderr
    assert not path.exists()
