# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest
import tt_umd
from helpers.logger import logger
from helpers.sim_host import READY_MARKER


def _proc_stat_alive(stat: bytes) -> bool:
    """True unless /proc/<pid>/stat shows a zombie or dead task.

    comm is the second field and may itself contain spaces or ')'.
    """
    close = stat.rfind(b")")
    if close < 0 or close + 2 >= len(stat):
        return False
    # "X" and "x" are both Dead in proc(5).
    return stat[close + 2 : close + 3] not in (b"Z", b"X", b"x")


def pid_alive(pid: int) -> bool:
    """True when pid is a live process.

    os.kill(pid, 0) succeeds for a zombie. Under xdist the controller does not
    reap the simulation host, so a host that dies mid-run stays in state Z and
    a kill-0 check would otherwise keep returning True.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes()
    except FileNotFoundError:
        return False
    except OSError:
        stat = None
    if stat is not None:
        return _proc_stat_alive(stat)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SimulationServer:
    """Runs sim_host.py, the UMD simulation host that test processes attach to."""

    HOST_SCRIPT = Path(__file__).with_name("sim_host.py")
    READY_TIMEOUT_S = 600
    POLL_INTERVAL_S = 2
    EXIT_TIMEOUT_S = 30
    NNG_SOCKET_ADDR = "NNG_SOCKET_ADDR"
    STALE_HOST_POLL_S = 0.5

    def __init__(self, simulator_path: str):
        self._simulator_path = simulator_path
        self._process: Optional[subprocess.Popen] = None
        self._pgid: Optional[int] = None
        self._log_path: Optional[str] = None
        self._emu_logs_baseline: set = set()
        self._log_read_offset = 0
        self._started_before = False
        self._server_directory: Optional[str] = None

    def start(self) -> None:
        self._kill_stale_hosts()
        tt_umd.SimulationConnector.prune_dead_servers()
        self._emu_logs_baseline = set(glob.glob(self.EMU_LOG_PATTERN))
        if not os.path.isdir(self._simulator_path):
            logger.error(
                "Simulator build path does not exist: {}", self._simulator_path
            )
            pytest.exit(returncode=1)

        missing_vars = [
            v
            for v in (self.NNG_SOCKET_ADDR, "NNG_SOCKET_LOCAL_PORT")
            if v not in os.environ
        ]
        if missing_vars:
            logger.error(
                "Required environment variable(s) not set: {}",
                ", ".join(missing_vars),
            )
            pytest.exit(returncode=1)

        self._log_path = os.path.join(os.getcwd(), "sim-host.log")
        if self._started_before and os.path.exists(self._log_path):
            self._log_read_offset = os.path.getsize(self._log_path)
            log_mode = "a"
        else:
            self._log_read_offset = 0
            log_mode = "w"
            self._started_before = True

        logger.info(
            "Starting UMD simulation host (simulator={}, "
            "{}={}, NNG_SOCKET_LOCAL_PORT={})...",
            self._simulator_path,
            self.NNG_SOCKET_ADDR,
            os.environ.get(self.NNG_SOCKET_ADDR, "<not set>"),
            os.environ.get("NNG_SOCKET_LOCAL_PORT", "<not set>"),
        )
        logger.info("Simulation host output: {}", self._log_path)

        self._server_directory = None
        with open(self._log_path, log_mode) as log_file:
            self._process = subprocess.Popen(
                [sys.executable, str(self.HOST_SCRIPT), self._simulator_path],
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
            "Waiting for the simulation host to become ready (timeout: {}s)...",
            self.READY_TIMEOUT_S,
        )
        shutdown_requested = False
        elapsed = 0
        while elapsed < self.READY_TIMEOUT_S:
            try:
                if self._process.poll() is not None:
                    log_tail = self._read_log_tail(50)
                    logger.error(
                        "Simulation host exited prematurely (code {}).\nLog output:\n{}",
                        self._process.returncode,
                        log_tail,
                    )
                    pytest.exit(returncode=1)

                self._server_directory = self._read_server_directory()
                if self._server_directory is not None:
                    logger.info(
                        "Simulation host ready (PID {}, took ~{}s), serving {}",
                        self._process.pid,
                        elapsed,
                        self._server_directory,
                    )
                    if shutdown_requested:
                        logger.info(
                            "Gracefully stopping the simulation host to release emulator..."
                        )
                        self.stop()
                        pytest.exit(
                            "Interrupted by user during simulation host startup.",
                            returncode=1,
                        )
                    return

                emu_errors = self._check_emulator_log()
                if emu_errors:
                    logger.error(
                        "Emulator reported errors during simulation host startup:\n{}",
                        emu_errors,
                    )
                    self.stop()
                    pytest.exit(returncode=1)

                time.sleep(self.POLL_INTERVAL_S)
            except KeyboardInterrupt:
                if not shutdown_requested:
                    shutdown_requested = True
                    logger.warning(
                        "Ctrl+C received, waiting for the simulation host to become ready "
                        "before shutting down (to release emulator resources)..."
                    )

            elapsed += self.POLL_INTERVAL_S
            if elapsed % 10 == 0:
                logger.info("    ... still waiting ({}s elapsed)", elapsed)

        log_tail = self._read_log_tail(50)
        if shutdown_requested:
            logger.error(
                "Simulation host did not become ready after Ctrl+C; "
                "giving up after {}s.\nLog output:\n{}",
                self.READY_TIMEOUT_S,
                log_tail,
            )
        else:
            logger.error(
                "Simulation host did not become ready within {}s.\nLog output:\n{}",
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

    def _read_server_directory(self) -> Optional[str]:
        if not self._log_path or not os.path.exists(self._log_path):
            return None
        try:
            with open(self._log_path, "rb") as f:
                f.seek(self._log_read_offset)
                new_data = f.read()
        except OSError:
            return None
        # Consume only whole lines.
        complete = new_data[: new_data.rfind(b"\n") + 1]
        self._log_read_offset += len(complete)
        for line in complete.decode(errors="replace").splitlines():
            marker, _, directory = line.partition(" ")
            if marker == READY_MARKER and directory.strip():
                return directory.strip()
        return None

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
            logger.info("Stopping simulation host (PID {})...", self._process.pid)
            try:
                self._process.stdin.write(b"exit\n")
                self._process.stdin.flush()
                self._process.stdin.close()
            except OSError:
                pass

            try:
                self._process.wait(timeout=self.EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Simulation host did not exit gracefully, "
                    "sending SIGTERM to process group {}...",
                    self._pgid,
                )
                self._kill_process_group(self._pgid, signal.SIGTERM)
                try:
                    self._process.wait(timeout=self.EXIT_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Process group {} did not terminate, sending SIGKILL...",
                        self._pgid,
                    )
                    self._kill_process_group(self._pgid, signal.SIGKILL)
                    self._process.wait()
            logger.info("Simulation host stopped.")

        self._process = None
        self._pgid = None
        self._server_directory = None

    @staticmethod
    def _kill_process_group(pgid: int, sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except (OSError, ProcessLookupError):
            pass

    def restart(self) -> None:
        logger.info("Restarting simulation host...")
        self.stop()
        self.start()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def ever_started(self) -> bool:
        return self._started_before

    @property
    def server_directory(self) -> Optional[str]:
        return self._server_directory

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    def _kill_stale_hosts(self) -> None:
        """Kill leftover hosts on the same simulator and NNG_SOCKET_ADDR (emulator slot)."""
        try:
            result = subprocess.run(
                ["pgrep", "-u", str(os.getuid()), "-f", self.HOST_SCRIPT.name],
                capture_output=True,
                text=True,
            )
        except OSError:
            return

        if result.returncode != 0 or not result.stdout.strip():
            return

        my_pid = os.getpid()
        nng_addr = os.environ.get(self.NNG_SOCKET_ADDR)
        nng_prefix = f"{self.NNG_SOCKET_ADDR}=".encode()
        stale_pids = []
        for line in result.stdout.strip().splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid == my_pid:
                continue
            try:
                cmdline_args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
                environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\x00")
            except OSError:
                continue
            if not any(
                arg.endswith(self.HOST_SCRIPT.name.encode()) for arg in cmdline_args
            ):
                continue
            if self._simulator_path.encode() not in cmdline_args:
                continue
            host_nng_addr = next(
                (
                    entry.split(b"=", 1)[1].decode(errors="replace")
                    for entry in environ
                    if entry.startswith(nng_prefix)
                ),
                None,
            )
            if host_nng_addr != nng_addr:
                continue
            stale_pids.append(pid)

        if not stale_pids:
            return

        logger.warning(
            "Found {} stale simulation host(s) for {} ({}={}): {}. Killing...",
            len(stale_pids),
            self._simulator_path,
            self.NNG_SOCKET_ADDR,
            nng_addr,
            stale_pids,
        )

        for pid in stale_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

        deadline = time.monotonic() + self.EXIT_TIMEOUT_S
        while time.monotonic() < deadline and any(
            Path(f"/proc/{pid}").exists() for pid in stale_pids
        ):
            time.sleep(self.STALE_HOST_POLL_S)

        for pid in stale_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
