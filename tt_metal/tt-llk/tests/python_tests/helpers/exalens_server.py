# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import shutil
import signal
import subprocess
import time
from typing import Optional

import pytest
from helpers.logger import logger


class ExalensServer:
    """Manages the tt-exalens server lifecycle for simulator-based test runs.

    Starts tt-exalens as a subprocess, waits for it to become ready by polling
    its output for the readiness pattern, and provides graceful shutdown.
    """

    READY_PATTERN = "[4B MODE]"
    READY_TIMEOUT_S = 600
    POLL_INTERVAL_S = 2

    def __init__(self, simulator_path: str, port: int):
        self._simulator_path = simulator_path
        self._port = port
        self._process: Optional[subprocess.Popen] = None
        self._pgid: Optional[int] = None
        self._log_path: Optional[str] = None
        self._emu_logs_baseline: set = set()
        self._log_read_offset = 0
        self._started_before = False

    def start(self) -> None:
        self._emu_logs_baseline = set(glob.glob(self.EMU_LOG_PATTERN))
        if not os.path.isdir(self._simulator_path):
            logger.error(
                "Simulator build path does not exist: {}", self._simulator_path
            )
            pytest.exit(returncode=1)

        if not shutil.which("tt-exalens"):
            logger.error("tt-exalens not found in PATH")
            pytest.exit(returncode=1)

        missing_vars = [
            v
            for v in ("NNG_SOCKET_ADDR", "NNG_SOCKET_LOCAL_PORT")
            if v not in os.environ
        ]
        if missing_vars:
            logger.error(
                "Required environment variable(s) not set: {}",
                ", ".join(missing_vars),
            )
            pytest.exit(returncode=1)

        self._log_path = os.path.join(os.getcwd(), "tt-exalens.log")
        if self._started_before and os.path.exists(self._log_path):
            self._log_read_offset = os.path.getsize(self._log_path)
            log_mode = "a"
        else:
            self._log_read_offset = 0
            log_mode = "w"
            self._started_before = True

        logger.info(
            "Starting tt-exalens server (port={}, simulator={}, "
            "NNG_SOCKET_ADDR={}, NNG_SOCKET_LOCAL_PORT={})...",
            self._port,
            self._simulator_path,
            os.environ.get("NNG_SOCKET_ADDR", "<not set>"),
            os.environ.get("NNG_SOCKET_LOCAL_PORT", "<not set>"),
        )
        logger.info("tt-exalens output: {}", self._log_path)

        with open(self._log_path, log_mode) as log_file:
            self._process = subprocess.Popen(
                [
                    "tt-exalens",
                    f"--port={self._port}",
                    "--server",
                    "-s",
                    self._simulator_path,
                ],
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            self._pgid = os.getpgid(self._process.pid)
        except OSError:
            self._pgid = self._process.pid

        self._wait_until_ready()

    EMU_LOG_PATTERN = "emu_*_.log"

    def _wait_until_ready(self) -> None:
        logger.info(
            "Waiting for tt-exalens to become ready (timeout: {}s)...",
            self.READY_TIMEOUT_S,
        )
        shutdown_requested = False
        elapsed = 0
        while elapsed < self.READY_TIMEOUT_S:
            try:
                if self._process.poll() is not None:
                    log_tail = self._read_log_tail(50)
                    logger.error(
                        "tt-exalens exited prematurely (code {}).\nLog output:\n{}",
                        self._process.returncode,
                        log_tail,
                    )
                    pytest.exit(returncode=1)

                if self._log_contains_ready_pattern():
                    logger.info(
                        "tt-exalens ready (PID {}, took ~{}s)",
                        self._process.pid,
                        elapsed,
                    )
                    if shutdown_requested:
                        logger.info(
                            "Gracefully stopping tt-exalens to release emulator..."
                        )
                        self.stop()
                        pytest.exit(
                            "Interrupted by user during tt-exalens startup.",
                            returncode=1,
                        )
                    return

                emu_errors = self._check_emulator_log()
                if emu_errors:
                    logger.error(
                        "Emulator reported errors during tt-exalens startup:\n{}",
                        emu_errors,
                    )
                    self.stop()
                    pytest.exit(returncode=1)

                time.sleep(self.POLL_INTERVAL_S)
            except KeyboardInterrupt:
                if not shutdown_requested:
                    shutdown_requested = True
                    logger.warning(
                        "Ctrl+C received — waiting for tt-exalens to become ready "
                        "before shutting down (to release emulator resources)..."
                    )

            elapsed += self.POLL_INTERVAL_S
            if elapsed % 10 == 0:
                logger.info("    ... still waiting ({}s elapsed)", elapsed)

        log_tail = self._read_log_tail(50)
        if shutdown_requested:
            logger.error(
                "tt-exalens did not become ready after Ctrl+C; "
                "giving up after {}s.\nLog output:\n{}",
                self.READY_TIMEOUT_S,
                log_tail,
            )
        else:
            logger.error(
                "tt-exalens did not become ready within {}s.\nLog output:\n{}",
                self.READY_TIMEOUT_S,
                log_tail,
            )
        self.stop()
        pytest.exit(returncode=1)

    EMU_ERROR_PATTERN = "zServer : ERROR"

    def _check_emulator_log(self) -> Optional[str]:
        """Check emulator logs created after start() for zServer ERROR lines."""
        new_logs = set(glob.glob(self.EMU_LOG_PATTERN)) - self._emu_logs_baseline
        if not new_logs:
            return None

        try:
            latest = max(new_logs, key=os.path.getmtime)
        except OSError:
            return None
        error_lines = []
        try:
            with open(latest, "r") as f:
                for line in f:
                    if self.EMU_ERROR_PATTERN in line:
                        error_lines.append(line.rstrip())
        except OSError:
            return None

        if error_lines:
            return f"(from {latest})\n" + "\n".join(error_lines)
        return None

    def _log_contains_ready_pattern(self) -> bool:
        if not self._log_path or not os.path.exists(self._log_path):
            return False
        try:
            with open(self._log_path, "r") as f:
                f.seek(self._log_read_offset)
                new_data = f.read()
                self._log_read_offset = f.tell()
                return self.READY_PATTERN in new_data
        except OSError:
            return False

    def _read_log_tail(self, lines: int = 30) -> str:
        if not self._log_path or not os.path.exists(self._log_path):
            return "<no log available>"
        try:
            with open(self._log_path, "r") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except OSError:
            return "<failed to read log>"

    def stop(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            logger.info("Stopping tt-exalens (PID {})...", self._process.pid)
            try:
                self._process.stdin.write(b"exit\n")
                self._process.stdin.flush()
                self._process.stdin.close()
            except OSError:
                pass

            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "tt-exalens did not exit gracefully, "
                    "sending SIGTERM to process group {}...",
                    self._pgid,
                )
                self._kill_process_group(self._pgid, signal.SIGTERM)
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Process group {} did not terminate, sending SIGKILL...",
                        self._pgid,
                    )
                    self._kill_process_group(self._pgid, signal.SIGKILL)
                    self._process.wait()
            logger.info("tt-exalens stopped.")

        self._process = None
        self._pgid = None

    @staticmethod
    def _kill_process_group(pgid: int, sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except (OSError, ProcessLookupError):
            pass

    def restart(self) -> None:
        logger.info("Restarting tt-exalens server...")
        self.stop()
        self.start()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def ever_started(self) -> bool:
        return self._started_before
