# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Entry point for the out-of-tree contract fixture.

The fixture under ``tests/out_of_tree/`` is a stand-in for a suite living in
another repository: it attaches to the harness the way
``docs/tests/getting_started.md`` §9 says to, and no other way. It therefore has
to be its own pytest rootdir — nested ``pytest_plugins`` is rejected by pytest,
and that constraint is itself part of the contract, so the fixture cannot simply
be collected into this run.

This wrapper bridges that: it is collected by the existing jobs (which all run
pytest from ``tests/python_tests/``) and drives the fixture as a subprocess with
its own rootdir. No CI configuration is needed, and a harness change that would
break an external consumer fails in the PR that makes it rather than during a
later submodule uplift.

A failure here is an API break. Fix the change, or land it with a deprecation
path — do not adjust the fixture until it passes again.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "out_of_tree"
TIMEOUT_SECONDS = 600


def test_out_of_tree_contract_fixture_passes():
    """Run the fixture suite as a real external consumer would.

    Always in ``--compile-producer`` mode: the contract covers configuration
    and the compile wiring, none of which needs silicon, so this behaves the
    same in the compile and run passes and on every runner.
    """
    assert FIXTURE_DIR.is_dir(), f"out-of-tree fixture missing at {FIXTURE_DIR}"

    env = dict(os.environ)
    # The fixture resolves LLK_HOME itself; keep whatever CI set, and drop
    # outer-run pytest state that would otherwise leak into the child.
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("PYTEST_XDIST_WORKER_COUNT", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--compile-producer",
            "-q",
            "--override-ini=addopts=",
            ".",
        ],
        cwd=FIXTURE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, (
        "The out-of-tree testing contract regressed. An external suite "
        "configuring the harness as documented in docs/tests/getting_started.md "
        "§9 no longer works.\n\n"
        f"--- fixture stdout ---\n{result.stdout}\n"
        f"--- fixture stderr ---\n{result.stderr}"
    )
