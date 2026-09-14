# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only regression tests: warmup artifacts must not certify real execution."""

import json
import hashlib
import subprocess
from pathlib import Path

import pytest

from tools.generic_op_to_factory import test_evidence
from tools.generic_op_to_factory.export_run import ExportError


@pytest.mark.parametrize("status", [2, 3, 4, 5, 137, 139])
def test_crash_or_incomplete_run_rejects_even_green_junit(tmp_path, status):
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuite><testcase name="warmup-passed"/></testsuite>')
    log = tmp_path / "run.log"
    log.write_text(f"SAFE_PYTEST_RAW_EXIT_CODE={status}\nSAFE_PYTEST_RESULT: FAIL\n")
    with pytest.raises(ExportError, match="did not complete normally"):  # allow-pytest.raises: host-only evidence guard
        test_evidence.verify(log, junit)


@pytest.mark.parametrize("text", ["", "SAFE_PYTEST_RAW_EXIT_CODE=0\nSAFE_PYTEST_RAW_EXIT_CODE=1\n"])
def test_missing_or_ambiguous_raw_status_rejected(tmp_path, text):
    log = tmp_path / "run.log"
    log.write_text(text)
    with pytest.raises(ExportError, match="exit status"):  # allow-pytest.raises: host-only evidence guard
        test_evidence.verify(log, tmp_path / "junit.xml")


def test_warmup_only_report_is_not_a_real_report(tmp_path):
    (tmp_path / "warmup.junit.xml").write_text("<testsuite/>")
    log = tmp_path / "run.log"
    log.write_text("SAFE_PYTEST_RAW_EXIT_CODE=1\n")
    with pytest.raises(ExportError, match="did not produce"):  # allow-pytest.raises: host-only evidence guard
        test_evidence.verify(log, tmp_path / "junit.xml")


@pytest.mark.parametrize("status", [0, 1])
def test_normal_outcomes_with_observed_route(tmp_path, status):
    route = {"mode": "native", "source": "package:source", "native": "package:native", "aliases": []}
    junit = tmp_path / "junit.xml"
    junit.write_text("<testsuite/>")
    log = tmp_path / "run.log"
    log.write_text(f"SAFE_PYTEST_RAW_EXIT_CODE={status}\nMIGRATION_ROUTE=" + json.dumps({**route, "calls": 2}) + "\n")
    evidence = test_evidence.verify(log, junit, route=route)
    assert evidence["pytest_exit_code"] == status
    assert evidence["route"]["calls"] == 2


@pytest.mark.parametrize("calls", [0, -1, False, "1"])
def test_zero_calls_rejected_even_when_tests_fail(tmp_path, calls):
    route = {"mode": "native"}
    junit = tmp_path / "junit.xml"
    junit.write_text("<testsuite/>")
    log = tmp_path / "run.log"
    log.write_text("SAFE_PYTEST_RAW_EXIT_CODE=1\nMIGRATION_ROUTE=" + json.dumps({**route, "calls": calls}) + "\n")
    with pytest.raises(ExportError, match="did not call"):  # allow-pytest.raises: host-only evidence guard
        test_evidence.verify(log, junit, route=route)


def test_old_runner_precompile_is_refused(tmp_path):
    runner = tmp_path / "runner.sh"
    runner.write_text("#!/bin/sh\n")
    with pytest.raises(ExportError, match="separate warmup"):  # allow-pytest.raises: host-only runner gate
        test_evidence.check_runner(runner, precompile=True, require_raw=False)
    test_evidence.check_runner(runner, precompile=False, require_raw=False)


def test_actual_warm_command_overrides_requested_junit_destination():
    script = Path(__file__).resolve().parents[3] / "scripts/run_safe_pytest.sh"
    source = script.read_text()
    assert 'pytest "${PYTEST_ARGS[@]}" --junitxml="${clog%.log}.junit.xml"' in source
    assert "SAFE_PYTEST_RAW_EXIT_CODE=$EXIT_CODE" in source
    assert source.index('wait "$PYTEST_TEE_PID"') < source.index('echo "SAFE_PYTEST_RAW_EXIT_CODE=$EXIT_CODE"')


def test_reviewed_runtime_tool_pins_match_shipped_files():
    root = Path(__file__).resolve().parents[3]
    assert hashlib.sha256((root / "scripts/run_safe_pytest.sh").read_bytes()).hexdigest() == test_evidence.RUNNER_SHA256
    adapter = root / "tools/generic_op_to_factory/native_adapter.py"
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == test_evidence.ADAPTER_SHA256


def test_real_runner_segment_waits_for_delayed_tee_and_starts_status_line(tmp_path):
    source = (Path(__file__).resolve().parents[3] / "scripts/run_safe_pytest.sh").read_text()
    segment = source[source.index("exec {PYTEST_TEE_FD}>") : source.index("# --- Handle result ---")]
    command = (
        'PYTEST_STDOUT_LOG="$1"\n'
        "PYTEST_CMD=(printf partial-output)\n"
        'tee() { sleep 0.05; command tee "$@"; }\n' + segment
    )
    result = subprocess.run(
        ["bash", "-c", command, "runner-segment", str(tmp_path / "stdout.log")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.startswith("partial-output\nSAFE_PYTEST_RAW_EXIT_CODE=0\n")


def test_runner_blocks_incomplete_evidence_when_descendant_retains_stdout(tmp_path):
    source = (Path(__file__).resolve().parents[3] / "scripts/run_safe_pytest.sh").read_text()
    segment = source[source.index("exec {PYTEST_TEE_FD}>") : source.index("# --- Handle result ---")]
    # Exercise the actual bounded-drain branch with a shorter test-only deadline.
    segment = segment.replace("timeout 10s tail", "timeout 0.05s tail")
    command = 'PYTEST_STDOUT_LOG="$1"\n' "PYTEST_CMD=(bash -c '(sleep 0.3) & printf partial-output')\n" + segment
    result = subprocess.run(
        ["bash", "-c", command, "runner-segment", str(tmp_path / "stdout.log")],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 3
    assert "execution evidence is incomplete" in result.stdout
    assert "SAFE_PYTEST_RAW_EXIT_CODE=" not in result.stdout


@pytest.mark.parametrize("status", [1, 2, 139])
def test_historical_runner_explicit_exit_is_checked(tmp_path, status):
    junit = tmp_path / "junit.xml"
    junit.write_text("<testsuite/>")
    log = tmp_path / "run.log"
    log.write_text(f"SAFE_PYTEST_RESULT: FAIL (pytest exit code: {status}; wrapper exit: 1)\n")
    if status == 1:
        assert test_evidence.verify(log, junit, require_raw=False)["pytest_exit_code"] == 1
    else:
        with pytest.raises(ExportError, match="did not complete normally"):  # allow-pytest.raises: evidence guard
            test_evidence.verify(log, junit, require_raw=False)


def test_historical_runner_missing_status_is_not_accepted(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("SAFE_PYTEST_RESULT: FAIL\n")
    with pytest.raises(ExportError, match="exit status"):  # allow-pytest.raises: evidence guard
        test_evidence.verify(log, tmp_path / "junit.xml", require_raw=False)


def test_runner_token_comment_is_not_warmup_routing(tmp_path):
    runner = tmp_path / "runner.sh"
    runner.write_text("# SAFE_PYTEST_RAW_EXIT_CODE= SAFE_PYTEST_WARMUP_JUNIT=\n")
    with pytest.raises(ExportError, match="separate warmup"):  # allow-pytest.raises: runner gate
        test_evidence.check_runner(runner, precompile=True)


def test_raw_marker_alone_does_not_admit_an_unreviewed_runner(tmp_path):
    runner = tmp_path / "runner.sh"
    runner.write_text('echo "SAFE_PYTEST_RAW_EXIT_CODE=$EXIT_CODE"\n')
    with pytest.raises(ExportError, match="reviewed version"):  # allow-pytest.raises: runtime tool version pin
        test_evidence.check_runner(runner, precompile=False)


def test_redirected_runner_is_refused(tmp_path):
    original = tmp_path / "original.sh"
    original.write_text("# SAFE_PYTEST_RAW_EXIT_CODE=\n")
    link = tmp_path / "runner.sh"
    link.symlink_to(original)
    with pytest.raises(ExportError, match="unredirected"):  # allow-pytest.raises: runner gate
        test_evidence.check_runner(link, precompile=False)
