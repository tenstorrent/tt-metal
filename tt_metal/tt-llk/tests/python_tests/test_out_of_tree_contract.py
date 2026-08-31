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
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "out_of_tree"
# Deliberately below the ambient pytest-timeout budget CI passes (--timeout=60
# in setup-and-test.yml and collect-test-durations.yml): our own timeout has to
# fire first, because ``_run_fixture`` turns it into a failure carrying the
# child's output, whereas pytest-timeout reports a bare kill. Measured cost of
# the child in CI: 7.1s on wormhole, so the headroom is ample.
TIMEOUT_SECONDS = 50

# Node ids the child must report as passing. Checking exact ids, from the junit
# report rather than from terminal text, is what makes this a real guard: a
# green child that collected the wrong module, or silently stopped running three
# tests, fails here. Keep in step with the fixture.
EXPECTED_FIXTURE_NODE_IDS = frozenset(
    {
        "test_exported_surface_matches_the_pinned_contract",
        "test_catalogue_aliases_import_both_ways",
        "test_names_the_harness_rebinds_are_resolved_at_call_time",
        "test_every_exported_name_resolves",
        "test_plugin_name_constant_matches_the_module_that_exists",
        "test_version_gate_accepts_compatible_and_rejects_the_rest",
        "test_no_consumer_module_reaches_past_the_facade",
        "test_plugin_supplies_the_pytest_hooks",
        "test_suite_local_package_imports_without_sys_path_edits",
        "test_helpers_resolves_to_the_harness_not_a_local_shadow",
        "test_out_of_tree_golden_registers",
        "test_register_golden_is_the_harness_decorator",
        "test_absolute_driver_path_is_accepted_and_keyed_safely",
        "test_llk_tree_include_roots_expands_one_arch_tree",
        "test_per_variant_search_dirs_are_accepted",
        "test_out_of_tree_driver_compiles",
        "test_per_variant_include_dirs_override_suite_wide_ones",
        "test_missing_test_name_is_rejected",
        "test_driver_path_that_breaks_the_include_directive_is_rejected",
    }
)


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
        report = Path(child_temp) / "fixture-report.xml"
        result = _run_fixture(child_temp, report)

        assert result.returncode == 0, (
            "The out-of-tree testing contract regressed. An external suite "
            "configuring the harness as documented in "
            "docs/tests/getting_started.md §9 no longer works.\n\n"
            f"--- fixture stdout ---\n{result.stdout}\n"
            f"--- fixture stderr ---\n{result.stderr}"
        )

        _assert_the_child_ran_the_fixture(report, result)


def _run_fixture(child_temp: str, report: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # The fixture resolves LLK_HOME itself; keep whatever CI set, and drop
    # outer-run pytest state that would otherwise leak into the child.
    # PYTEST_ADDOPTS matters as much as the xdist pair: pytest splices it
    # straight into the child argv, and ``--override-ini=addopts=`` clears the
    # ini key, not the environment variable.
    for leaked in (
        "PYTEST_CURRENT_TEST",
        "PYTEST_XDIST_WORKER",
        "PYTEST_XDIST_WORKER_COUNT",
        "PYTEST_ADDOPTS",
    ):
        env.pop(leaked, None)
    # Private artefact tree, locks and stimuli cache included — see the
    # docstring. Both names matter: RUNNER_TEMP wins when it is set.
    env["RUNNER_TEMP"] = child_temp
    env["TMPDIR"] = child_temp

    argv = [
        sys.executable,
        "-m",
        "pytest",
        "--compile-producer",
        "-q",
        "--override-ini=addopts=",
        f"--junitxml={report}",
        ".",
    ]

    # start_new_session so the whole process group can be killed on timeout:
    # the child fans compiles out across a ThreadPoolExecutor, and killing only
    # the direct child would leave those compilers running on a shared runner.
    process = subprocess.Popen(
        argv,
        cwd=FIXTURE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(
            f"the out-of-tree fixture did not finish within {TIMEOUT_SECONDS}s "
            "(it normally takes a few seconds; a hang here usually means the "
            "child is waiting on a lock or a compile).\n\n"
            f"--- fixture stdout ---\n{stdout}\n"
            f"--- fixture stderr ---\n{stderr}"
        )

    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _assert_the_child_ran_the_fixture(
    report: Path, result: subprocess.CompletedProcess
) -> None:
    """A green child is not enough; it has to be green for the right reasons.

    Read from the junit report, not the terminal output: ``log_cli`` and
    verbosity change how pytest lays out ``PASSED``, so counting text would be a
    guess about formatting rather than a statement about what ran.
    """
    assert report.is_file(), (
        f"the fixture produced no junit report at {report}; it may not have "
        f"reached collection.\n\n{result.stdout}"
    )

    root = ET.parse(report).getroot()
    ran = {
        case.get("name", "").split("[")[0]
        for case in root.iter("testcase")
        if not any(case.iter("failure")) and not any(case.iter("error"))
    }

    missing = sorted(EXPECTED_FIXTURE_NODE_IDS - ran)
    assert not missing, (
        f"the fixture did not run {missing}. Either the child collected the "
        "wrong module, those tests stopped running, or they were renamed "
        f"without updating EXPECTED_FIXTURE_NODE_IDS.\n\n{result.stdout}"
    )
