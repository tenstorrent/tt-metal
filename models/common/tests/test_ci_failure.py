# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

from models.common import ci_failure
from models.common.ci_failure import PerfGateFailure, assert_perf_gate, record_deterministic_failure

REPO = Path(__file__).resolve().parents[3]


def records(directory):
    return sorted(p.read_text() for p in Path(directory).glob("*.txt"))


def test_no_op_without_records_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(ci_failure.RECORDS_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    record_deterministic_failure("TPS GATE FAIL")
    assert list(tmp_path.iterdir()) == []


def test_records_one_line_per_failure_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv(ci_failure.RECORDS_ENV, str(tmp_path / "records"))
    record_deterministic_failure("TPS GATE FAIL:\n  21,385 outside [17,698, 19,560]")
    record_deterministic_failure("")
    assert records(tmp_path / "records") == [
        "TPS GATE FAIL: 21,385 outside [17,698, 19,560]\n",
        "deterministic failure (no reason given)\n",
    ]
    assert not list((tmp_path / "records").glob(".*.tmp")), "temporary files must be renamed into place"


def test_unwritable_dir_never_raises(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(ci_failure.RECORDS_ENV, str(blocker / "records"))
    record_deterministic_failure("TPS GATE FAIL")
    assert "not recorded" in capsys.readouterr().err


def test_assert_perf_gate(expect_error):
    assert_perf_gate(True, "unused")
    with expect_error(PerfGateFailure, "outside band"):
        assert_perf_gate(False, "outside band")
    # Existing callers that expect AssertionError keep working.
    with expect_error(AssertionError, "outside band"):
        assert_perf_gate(False, "outside band")


def test_cli(tmp_path):
    env = {**os.environ, ci_failure.RECORDS_ENV: str(tmp_path), "PYTHONPATH": str(REPO)}
    subprocess.run([sys.executable, "-m", "models.common.ci_failure", "TTFT", "GATE", "FAIL"], env=env, check=True)
    assert records(tmp_path) == ["TTFT GATE FAIL\n"]
