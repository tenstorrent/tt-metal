# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Execute the diag suite as a subprocess.

This replaces the previous docker-in-docker model: when the whole health check
runs inside the tt-metal image, the diag suite is a sibling script we invoke
directly rather than a nested container.

The command differs by launch mode (this is one of the two spots that diverge
between the bare-Slurm and the Kubernetes/orchestration deployments):

    slurm          -> ``python3 diag_runner.py --tier <tier> --output <report>``
    orchestration  -> ``bash run_diag.sh <tier> --output <report>``

Both share the same subprocess core: we keep a hard wall-clock budget and kill
the whole process group (gtest / tt-smi children included) when the timeout
fires, since the diag suite drives destructive resets and long gtests that can
hang.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Grace period between SIGTERM and SIGKILL when killing a timed-out diag run.
_TERM_GRACE_SECONDS = 10


def default_diag_runner() -> Path:
    """Locate diag_runner.py — two levels up from this module
    (test_infrastructure/utils/ -> health_check_test_suite/diag_runner.py)."""
    return Path(__file__).resolve().parents[2] / "diag_runner.py"


def default_run_diag_script() -> Path:
    """Locate run_diag.sh (the bash wrapper used in orchestration mode)."""
    return Path(__file__).resolve().parents[2] / "run_diag.sh"


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the child's process group (gtest / tt-smi children)."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=_TERM_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            continue


def _diag_env(tt_metal_path: Path | None) -> dict:
    """Environment shared by both launch modes."""
    env = os.environ.copy()
    if tt_metal_path is not None:
        root = str(tt_metal_path)
        lib_dir = str(tt_metal_path / "build" / "lib")
        env["TT_METAL_HOME"] = root
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (lib_dir, env.get("LD_LIBRARY_PATH"))))
    # tt-smi often lives at ~/.local/bin or /usr/local/bin but isn't on PATH in
    # non-login SSH sessions.
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), str(Path.home() / ".local" / "bin"), "/usr/local/bin"])
    return env


def _build_slurm_command(
    tier: str,
    report_path: Path,
    tt_metal_path: Path | None,
    diag_runner: Path | None,
    extra_args: list[str] | None,
) -> tuple[list[str] | None, str]:
    """Bare-Slurm: invoke diag_runner.py directly with the current interpreter."""
    runner = Path(diag_runner) if diag_runner else default_diag_runner()
    if not runner.is_file():
        return None, f"diag runner not found: {runner}"
    cmd = [sys.executable, str(runner), "--tier", tier, "--output", str(report_path)]
    if tt_metal_path is not None:
        cmd += ["--tt-metal-path", str(tt_metal_path)]
    if extra_args:
        cmd += list(extra_args)
    return cmd, ""


def _build_orchestration_command(
    tier: str,
    report_path: Path,
    tt_metal_path: Path | None,
    extra_args: list[str] | None,
) -> tuple[list[str] | None, str]:
    """Orchestration (k8s): invoke the sibling run_diag.sh bash wrapper.

    run_diag.sh takes the tier positionally and forwards the rest to
    diag_runner.py, so ``bash run_diag.sh <tier> --output <report>`` is
    equivalent to the Slurm command but goes through the shell wrapper the k8s
    worker has always used.
    """
    script = default_run_diag_script()
    if not script.is_file():
        return None, f"diag test script not found: {script}"
    cmd = ["bash", str(script), tier, "--output", str(report_path)]
    if tt_metal_path is not None:
        cmd += ["--tt-metal-path", str(tt_metal_path)]
    if extra_args:
        cmd += list(extra_args)
    return cmd, ""


def _execute(cmd: list[str], env: dict, timeout_seconds: int, results_dir: Path) -> tuple[int, str, Path | None]:
    """Run cmd with a hard timeout, killing the whole process group on expiry.

    Returns ``(exit_code, combined_console_output, results_dir | None)``. The
    diag tool exits 1 on FAIL and 0 on PASS/WARN; that status is propagated
    unchanged. A timeout is reported as exit code 137 (SIGKILL).
    """
    log.info("Running diag suite: %s", " ".join(cmd))
    log.info("Results dir: %s", results_dir)
    log.info("Timeout: %d seconds", timeout_seconds)

    started_at = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            # New session so the diag suite's gtest/tt-smi children share a
            # process group we can signal as a unit on timeout.
            start_new_session=True,
        )
    except OSError as exc:
        log.error("Failed to start diag suite: %s", exc)
        return 1, f"Failed to start diag suite: {exc}", None

    try:
        output, _ = proc.communicate(timeout=timeout_seconds)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started_at
        log.error(
            "Diag suite timed out after %.0f seconds (budget %d); killing",
            elapsed,
            timeout_seconds,
        )
        _kill_process_group(proc)
        # Keep whatever the suite emitted before it hung.
        try:
            output, _ = proc.communicate(timeout=_TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            output = ""
        exit_code = 137

    return exit_code, output or "", results_dir


def run_diag_subprocess(
    tier: str,
    timeout_seconds: int,
    results_dir: Path,
    *,
    launch_mode: str = "slurm",
    diag_runner: Path | None = None,
    tt_metal_path: Path | None = None,
    extra_args: list[str] | None = None,
) -> tuple[int, str, Path | None]:
    """Run the diag suite, writing its JSON report + per-test logs into results_dir.

    ``launch_mode`` selects how the diag suite is invoked (``slurm`` runs
    ``diag_runner.py`` directly; ``orchestration`` runs the sibling ``run_diag.sh``
    bash wrapper). Both write the report to ``<results_dir>/diag_report.json`` and
    drop per-test gtest logs under ``<results_dir>/logs/`` straight on the host
    filesystem, so partial results survive a timeout/kill.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "diag_report.json"
    report_path.unlink(missing_ok=True)

    if launch_mode == "orchestration":
        cmd, err = _build_orchestration_command(tier, report_path, tt_metal_path, extra_args)
    else:
        cmd, err = _build_slurm_command(tier, report_path, tt_metal_path, diag_runner, extra_args)

    if cmd is None:
        log.error("%s", err)
        return 1, err, None

    return _execute(cmd, _diag_env(tt_metal_path), timeout_seconds, results_dir)
