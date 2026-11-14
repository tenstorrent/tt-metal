# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line behavior of the multihost rankfile helper; no device needed."""

import subprocess
import sys
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "sources"
    / "examples"
    / "python"
    / "multihost"
    / "pipeline_parallel_training"
    / "make_rankfile.py"
)


def test_unrecognized_hostname_warns_on_stderr_and_is_placed_last():
    result = subprocess.run(
        [sys.executable, "-I", str(_SCRIPT), "-n", "1"],
        input="c09u08\nc09u02\nunexpected-host\nc10u02\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "rank 0=c09u08 slot=0",
        "rank 1=c09u02 slot=0",
        "rank 2=c10u02 slot=0",
        "rank 3=unexpected-host slot=0",
    ]
    assert result.stderr == (
        "warning: hostname 'unexpected-host' does not match pattern '<digits>u<2 digits>'; "
        "placing last in canonical order\n"
    )
