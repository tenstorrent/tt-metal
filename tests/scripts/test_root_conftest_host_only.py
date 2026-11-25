# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The root conftest lets a host-only test collect and run without ttnn, so without touching any device.

Every device access from Python goes through ttnn. The test runs a host-only probe in a child pytest whose
``ttnn`` import is blocked: if the root conftest imported ttnn at load, in a hook, or in an autouse fixture
(for example to detect the arch for ``--tt-arch``), the child fails.
"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Set only in the child pytest's environment. It tells the probe that it must be running in the blocked process.
_PROBE_ENV = "TT_ROOT_CONFTEST_HOST_ONLY_PROBE"

_BLOCK_TTNN_AND_RUN_PYTEST = (
    "import sys; sys.modules['ttnn'] = None; import pytest; sys.exit(pytest.main(sys.argv[1:]))"
)


def test_host_only_probe(expect_error):
    """A host-only test that uses a root-conftest fixture; run alone by the test below."""
    if os.environ.get(_PROBE_ENV):
        # The block lives only in the child's own interpreter. A process that does not inherit it, such as an
        # xdist worker, would let the probe pass even if the root conftest imported ttnn.
        assert sys.modules.get("ttnn", "not blocked") is None, "the probe must run in the process that blocks ttnn"
    with expect_error(ValueError, "host-only"):
        int("host-only")


def test_host_only_test_runs_without_ttnn(tmp_path):
    probe = f"{Path(__file__).relative_to(_REPO_ROOT)}::test_host_only_probe"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _BLOCK_TTNN_AND_RUN_PYTEST,
            probe,
            "-p",
            "no:cacheprovider",
            f"--junitxml={tmp_path / 'probe.xml'}",
        ],
        cwd=_REPO_ROOT,
        # PYTEST_ADDOPTS may carry -n N (CI sim legs set it), which would move the probe into xdist workers.
        env={**{k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}, _PROBE_ENV: "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 passed" in output, output
