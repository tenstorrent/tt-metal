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
import tempfile
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "out_of_tree"
TIMEOUT_SECONDS = 600

# Tests the child must actually have run. Without this the wrapper passes on any
# green child — including one that collected the wrong module and ran a single
# unrelated test, which is exactly how a mis-copied fixture file went unnoticed.
EXPECTED_FIXTURE_TESTS = (
    "test_exported_surface_matches_the_pinned_contract",
    "test_out_of_tree_driver_compiles",
    "test_per_variant_include_dirs_override_suite_wide_ones",
)
MIN_FIXTURE_TESTS = 15


def test_out_of_tree_contract_fixture_passes():
    """Run the fixture suite as a real external consumer would.

    Always in ``--compile-producer`` mode: the contract covers configuration
    and the compile wiring, none of which needs silicon, so this behaves the
    same in the compile and run passes and on every runner.

    The child gets its **own artefact tree**, and this is not optional. A
    producer-mode session wipes ``ARTEFACTS_DIR`` on startup
    (``TestConfig.setup_mode``), which is safe for a normal run — the xdist
    controller does it before workers spawn — but lethal here: this test runs
    *inside* a session whose workers are concurrently building into that same
    tree, so an un-isolated child deletes their variant directories mid-build.
    The symptom is unrelated tests failing with ``ld: cannot open output file
    .../elf/unpack.elf: No such file or directory``.

    ``resolve_artefacts_path()`` reads ``RUNNER_TEMP`` when set and otherwise
    falls back to ``tempfile.gettempdir()``, so both are redirected: the former
    covers GitHub Actions, the latter local runs. Isolation also makes this
    hermetic — the child never reads or writes the outer run's cache.
    """
    assert FIXTURE_DIR.is_dir(), f"out-of-tree fixture missing at {FIXTURE_DIR}"

    with tempfile.TemporaryDirectory(prefix="oot-contract-") as child_temp:
        result = _run_fixture(child_temp)

    assert result.returncode == 0, (
        "The out-of-tree testing contract regressed. An external suite "
        "configuring the harness as documented in docs/tests/getting_started.md "
        "§9 no longer works.\n\n"
        f"--- fixture stdout ---\n{result.stdout}\n"
        f"--- fixture stderr ---\n{result.stderr}"
    )

    _assert_the_child_ran_the_fixture(result)


def _run_fixture(child_temp: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # The fixture resolves LLK_HOME itself; keep whatever CI set, and drop
    # outer-run pytest state that would otherwise leak into the child.
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("PYTEST_XDIST_WORKER_COUNT", None)
    # Private artefact tree, locks and stimuli cache included — see the
    # docstring. Both names matter: RUNNER_TEMP wins when it is set.
    env["RUNNER_TEMP"] = child_temp
    env["TMPDIR"] = child_temp

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--compile-producer",
            "-v",
            "--override-ini=addopts=",
            ".",
        ],
        cwd=FIXTURE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )


def _assert_the_child_ran_the_fixture(result: subprocess.CompletedProcess) -> None:
    """A green child is not enough; it has to be green for the right reasons."""
    missing = [name for name in EXPECTED_FIXTURE_TESTS if name not in result.stdout]
    assert not missing, (
        f"the fixture run did not include {missing}. The child may have "
        "collected the wrong module, or those tests were renamed without "
        f"updating EXPECTED_FIXTURE_TESTS.\n\n{result.stdout}"
    )

    ran = result.stdout.count(" PASSED")
    assert ran >= MIN_FIXTURE_TESTS, (
        f"only {ran} fixture tests ran, expected at least {MIN_FIXTURE_TESTS}; "
        f"the contract suite is not being exercised.\n\n{result.stdout}"
    )
