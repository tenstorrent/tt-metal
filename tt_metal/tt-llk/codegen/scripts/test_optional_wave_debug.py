# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
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
    attempt: int = 4,
    fsdb: str | None = None,
    arch: str | None = None,
    scope: str | None = None,
    environment: dict[str, str] | None = None,
) -> wave.WaveDebugOutcome:
    return wave.run_optional_wave_debug(
        log_dir=tmp_path / "run",
        cycle=2,
        attempt=attempt,
        failure_kind="hang",
        fsdb=fsdb,
        launcher=launcher,
        timeout_seconds=10,
        arch=arch,
        scope=scope,
        environment=environment or {},
    )


def _argv_recording_launcher(tmp_path: Path) -> tuple[Path, Path]:
    """Return a fake private debugger that records its argv and emits evidence."""

    launcher = tmp_path / "argv_recording_debugger.py"
    argv_path = tmp_path / "argv.json"
    launcher.write_text(
        f"""\
import json
import pathlib
import sys

pathlib.Path({str(argv_path)!r}).write_text(json.dumps(sys.argv[1:]))
out = pathlib.Path(sys.argv[sys.argv.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "evidence.json").write_text(json.dumps({{
    "status": "inconclusive",
    "findings": []
}}))
"""
    )
    return launcher, argv_path


def test_first_three_attempts_skip_waves_without_touching_outputs(tmp_path):
    launcher = tmp_path / "must-not-run.py"
    marker = tmp_path / "launcher-ran"
    launcher.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
    )

    for attempt in (1, 2, 3):
        outcome = _run(tmp_path, launcher=launcher, attempt=attempt)
        assert outcome.status == "skipped"
        assert "log/source-only phase" in outcome.summary

    assert not marker.exists()
    assert not (tmp_path / "run").exists()


def test_only_attempts_four_and_five_are_wave_debug_eligible():
    assert [wave.wave_debug_eligible(attempt) for attempt in range(1, 6)] == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_attempt_outside_the_simulator_budget_skips_without_touching_outputs(tmp_path):
    for attempt in (0, 6):
        assert not wave.wave_debug_eligible(attempt)
        outcome = _run(tmp_path, launcher=tmp_path / "missing.py", attempt=attempt)
        assert outcome.status == "skipped"
        assert "outside the 1-5 budget" in outcome.summary

    assert not (tmp_path / "run").exists()


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


def test_remote_backend_accepts_an_fsdb_that_is_not_visible_locally(tmp_path):
    remote_fsdb = "/remote/only/failure.fsdb"
    assert not Path(remote_fsdb).exists()

    outcome = _run(
        tmp_path,
        launcher=tmp_path / "missing-launcher.py",
        fsdb=remote_fsdb,
        environment={"SSH_MACHINE_NAME": "soc-l-12"},
    )

    # The path must survive resolution and fail later on the missing launcher,
    # not be rejected as "does not exist" by a local stat.
    assert outcome.status == "unavailable"
    assert "launcher was not found" in outcome.summary
    assert remote_fsdb in (tmp_path / "run" / "agent_tester_cycle2.md").read_text()


def test_remote_fsdb_is_discovered_from_run_log_without_a_local_stat(tmp_path):
    remote_fsdb = "/remote/only/from-log.fsdb"
    run_log = tmp_path / "run" / "test_logs_cycle2" / "run.log"
    run_log.parent.mkdir(parents=True)
    run_log.write_text(f"simulator wrote waveform: {remote_fsdb}\n")

    outcome = _run(
        tmp_path,
        launcher=tmp_path / "missing-launcher.py",
        environment={"LLK_DEBUG_HOST": "soc-l-12"},
    )

    assert outcome.status == "unavailable"
    assert "launcher was not found" in outcome.summary
    assert remote_fsdb in (tmp_path / "run" / "agent_tester_cycle2.md").read_text()


def test_local_backend_still_rejects_a_missing_fsdb(tmp_path):
    outcome = _run(
        tmp_path,
        launcher=tmp_path / "missing-launcher.py",
        fsdb="/no/such/failure.fsdb",
    )

    assert outcome.status == "unavailable"
    assert "does not exist or is not readable" in outcome.summary


def test_arch_and_scope_are_forwarded_only_when_requested(tmp_path):
    fsdb = tmp_path / "failure.fsdb"
    fsdb.write_text("fixture")
    launcher, argv_path = _argv_recording_launcher(tmp_path)

    assert _run(tmp_path, launcher=launcher, fsdb=str(fsdb)).status == "inconclusive"
    default_argv = json.loads(argv_path.read_text())
    assert "--arch" not in default_argv and "--scope" not in default_argv

    # Pure passthrough: the private tool owns which values are valid.
    _run(
        tmp_path,
        launcher=launcher,
        fsdb=str(fsdb),
        arch="quasar",
        scope="gen_y[2].gen_x[3]",
    )
    override_argv = json.loads(argv_path.read_text())
    assert override_argv[override_argv.index("--arch") + 1] == "quasar"
    assert override_argv[override_argv.index("--scope") + 1] == "gen_y[2].gen_x[3]"


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
        tmp_path / "run" / "test_logs_cycle2" / "wave_debug_attempt4" / "evidence.json"
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
            "4",
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


def test_cli_skips_wave_debug_during_first_three_attempts(tmp_path):
    log_dir = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log-dir",
            str(log_dir),
            "--cycle",
            "1",
            "--attempt",
            "3",
            "--failure-kind",
            "hang",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "WAVE_DEBUG status=skipped" in result.stdout
    assert not log_dir.exists()
