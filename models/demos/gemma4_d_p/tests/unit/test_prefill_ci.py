# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "outcome, expected_code",
    [("passed", 0), ("failed", 7), ("skipped", 1), ("missing_inputs", 3)],
)
def test_ci_launcher(outcome, expected_code, tmp_path):
    repo = Path(__file__).resolve().parents[5]
    commands = tmp_path / "bin"
    commands.mkdir()
    python = commands / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        + r"""
import os
import sys
from pathlib import Path

outcome = os.environ["TEST_OUTCOME"]
if sys.argv[1] == "-":
    sys.stdin.read()
    sys.exit(3 if outcome == "missing_inputs" else 0)

options = dict(arg.split("=", 1) for arg in sys.argv if "=" in arg)
base = Path(options["--basetemp"])
Path(os.environ["SCRATCH_RECORD"]).write_text(str(base.parent))
case = base / "test_prefill_migration_mock_250"
case.mkdir(parents=True)
(case / "runner.log").write_text("runner output\n")
(case / "producer.log").write_text("producer output\n")
(case / "device_map.json").write_text("{}")
(case / "table.pb").write_bytes(b"address table")
Path(options["--junitxml"]).write_text("<testsuites/>")
if outcome == "passed":
    (case / "gemma4_slot0.json").write_text("{}")
sys.exit(7 if outcome == "failed" else 0)
"""
    )
    python.chmod(0o755)
    summaries = tmp_path / "summaries"
    scratch_record = tmp_path / "scratch.txt"
    env = dict(
        os.environ,
        PATH=f"{commands}:{os.environ['PATH']}",
        TT_METAL_HOME=str(repo),
        PREFILL_SUMMARIES=str(summaries),
        TEST_OUTCOME=outcome,
        SCRATCH_RECORD=str(scratch_record),
    )
    result = subprocess.run(
        ["bash", str(repo / "models/demos/gemma4_d_p/tests/ci/run_mock_256k.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    artifacts = summaries / "gemma4_mock256k"
    if outcome == "missing_inputs":
        assert not scratch_record.exists()
        assert not list(artifacts.iterdir())
    else:
        assert (artifacts / "runner.log").read_text() == "runner output\n"
        assert (artifacts / "producer.log").read_text() == "producer output\n"
        assert (artifacts / "junit.xml").is_file()
        assert (artifacts / "device_map.json").is_file()
        assert not (artifacts / "table.pb").exists()
        assert (artifacts / "gemma4_slot0.json").is_file() == (outcome == "passed")
        assert not Path(scratch_record.read_text()).exists()
    if outcome == "skipped":
        assert "Missing PCC report" in result.stderr
