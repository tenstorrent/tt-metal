# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "optional_wave_debug", Path(__file__).parent / "optional_wave_debug.py"
)
assert _SPEC and _SPEC.loader
wave = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = wave
_SPEC.loader.exec_module(wave)
SCRIPT = Path(__file__).parent / "optional_wave_debug.py"


def _run(
    tmp_path: Path,
    *,
    launcher: Path,
    fsdb: str | None = None,
    environment: dict[str, str] | None = None,
) -> wave.WaveDebugOutcome:
    return wave.run_optional_wave_debug(
        log_dir=tmp_path / "run",
        cycle=2,
        attempt=3,
        failure_kind="hang",
        fsdb=fsdb,
        launcher=launcher,
        timeout_seconds=10,
        environment=environment or {},
    )


def test_missing_fsdb_is_recorded_and_nonfatal(tmp_path):
    outcome = _run(tmp_path, launcher=tmp_path / "missing-launcher.py")

    assert outcome.status == "unavailable"
    log = (tmp_path / "run" / "agent_tester_cycle2.md").read_text()
    assert "Status: `unavailable`" in log
    assert "no FSDB was supplied" in log
    assert "continue the normal tester/refiner flow" in log


def test_missing_private_launcher_is_recorded_and_nonfatal(tmp_path):
    fsdb = tmp_path / "failure.fsdb"
    fsdb.write_text("fixture")

    outcome = _run(
        tmp_path,
        launcher=tmp_path / "missing-launcher.py",
        fsdb=str(fsdb),
    )

    assert outcome.status == "unavailable"
    log = (tmp_path / "run" / "agent_tester_cycle2.md").read_text()
    assert "launcher was not found" in log
    assert str(fsdb) in log


def test_fsdb_can_be_discovered_from_existing_run_log(tmp_path):
    fsdb = tmp_path / "from-log.fsdb"
    fsdb.write_text("fixture")
    run_log = tmp_path / "run" / "test_logs_cycle2" / "run.log"
    run_log.parent.mkdir(parents=True)
    run_log.write_text(f"simulator wrote waveform: {fsdb}\n")

    outcome = _run(tmp_path, launcher=tmp_path / "missing-launcher.py")

    assert outcome.status == "unavailable"
    assert "launcher was not found" in outcome.summary
    assert str(fsdb) in (tmp_path / "run" / "agent_tester_cycle2.md").read_text()


def test_successful_private_diagnosis_uses_existing_run_outputs(tmp_path):
    fsdb = tmp_path / "failure.fsdb"
    fsdb.write_text("fixture")
    launcher = tmp_path / "fake_private_debugger.py"
    launcher.write_text(
        """\
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[sys.argv.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "evidence.json").write_text(json.dumps({
    "status": "findings",
    "tool_source": {
        "repository": "llk_code_gen",
        "revision": "a" * 40,
        "dirty": False
    },
    "findings": [{
        "severity": "critical",
        "classification": "synthetic_terminal_stall",
        "summary": "Synthetic deterministic finding."
    }]
}))
"""
    )

    outcome = _run(tmp_path, launcher=launcher, fsdb=str(fsdb))

    assert outcome.status == "findings"
    assert outcome.evidence_path == (
        tmp_path / "run" / "test_logs_cycle2" / "wave_debug_attempt3" / "evidence.json"
    )
    log = (tmp_path / "run" / "agent_tester_cycle2.md").read_text()
    assert "synthetic_terminal_stall" in log
    assert f"llk_code_gen@{'a' * 40}" in log
    assert "Pipeline action: continue" in log


def test_private_debugger_failure_is_recorded_without_raising(tmp_path):
    fsdb = tmp_path / "failure.fsdb"
    fsdb.write_text("fixture")
    launcher = tmp_path / "failing_private_debugger.py"
    launcher.write_text(
        "import sys\nprint('private tool unavailable', file=sys.stderr)\nsys.exit(2)\n"
    )

    outcome = _run(tmp_path, launcher=launcher, fsdb=str(fsdb))

    assert outcome.status == "failed"
    assert "exited 2" in outcome.summary
    log = (tmp_path / "run" / "agent_tester_cycle2.md").read_text()
    assert "Status: `failed`" in log
    assert "continue the normal tester/refiner flow" in log


def test_cli_continues_when_private_checkout_is_missing(tmp_path):
    fsdb = tmp_path / "failure.fsdb"
    fsdb.write_text("fixture")
    log_dir = tmp_path / "run"
    environment = {
        **os.environ,
        "LLK_CODEGEN_PRIVATE_ROOT": str(tmp_path / "missing-private-checkout"),
    }

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log-dir",
            str(log_dir),
            "--cycle",
            "1",
            "--attempt",
            "1",
            "--failure-kind",
            "hang",
            "--fsdb",
            str(fsdb),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    log = (log_dir / "agent_tester_cycle1.md").read_text()
    assert "Status: `failed`" in log
    assert "LLK waveform debugger is not available" in log
    assert "continue the normal tester/refiner flow" in log
