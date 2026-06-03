# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import signal
import shutil
import subprocess
import time
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger


class ResetUtil:
    SUPPORTED_ARCHS = {"wormhole_b0", "blackhole"}

    def __init__(self, arch: str):
        if arch not in self.SUPPORTED_ARCHS:
            raise ValueError(f"SWEEPS: Unsupported Architecture for TT-SMI Reset: {arch}")

        self.arch = arch
        self.command, self.args = self._find_command()
        # Retry policy (overridable via env for tuning per runner).
        self.reset_attempts = max(1, int(os.getenv("TT_SMI_RESET_ATTEMPTS", "3")))
        self.reset_backoff_seconds = max(0, int(os.getenv("TT_SMI_RESET_BACKOFF_SECONDS", "30")))

    def _find_command(self):
        custom_command = os.getenv("TT_SMI_RESET_COMMAND")
        if custom_command:
            parts = custom_command.split()
            command, args = parts[0], parts[1:]
            if not shutil.which(command):
                raise FileNotFoundError(f"SWEEPS: Custom command not found: {command}")
            return command, args

        executable = shutil.which("tt-smi")
        if not executable:
            raise FileNotFoundError("SWEEPS: Unable to locate tt-smi executable")

        logger.info(f"tt-smi executable: {executable}")
        return executable, ["-r"]

    def _self_and_ancestors(self):
        """PIDs of this process and its ancestor chain (never to be killed)."""
        pids = set()
        pid = os.getpid()
        while pid and pid > 1 and pid not in pids:
            pids.add(pid)
            try:
                with open(f"/proc/{pid}/stat") as f:
                    # ppid is the 4th field, but the comm (2nd field) may contain
                    # spaces/parens — parse after the closing ')'.
                    data = f.read()
                    pid = int(data[data.rindex(")") + 1 :].split()[1])
            except Exception:
                break
        return pids

    def _device_holder_pids(self):
        """PIDs (excluding self + ancestors) holding a /dev/tenstorrent fd.

        A vector that hangs in a device call is SIGKILLed by the runner before
        the reset, but a process stuck in an uninterruptible (D-state) driver
        call keeps the device claimed until that call returns; tt-smi then can't
        reset it and returns non-zero. Best-effort kill of such leftovers (and
        a short wait) lets the reset proceed.
        """
        protected = self._self_and_ancestors()
        holders = []
        try:
            proc_entries = os.listdir("/proc")
        except OSError:
            return holders
        for entry in proc_entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid in protected:
                continue
            fd_dir = f"/proc/{entry}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue
            for fd in fds:
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if "tenstorrent" in target:
                    holders.append(pid)
                    break
        return holders

    def _free_device(self):
        """Best-effort: SIGKILL stray processes still holding the device."""
        holders = self._device_holder_pids()
        if not holders:
            return
        logger.warning(f"SWEEPS: killing stray process(es) holding /dev/tenstorrent before reset: {holders}")
        for pid in holders:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        # Give the kernel a moment to tear down the killed processes' handles.
        time.sleep(2)

    def reset(self):
        """Execute the reset command with cleanup + bounded backoff retries.

        glx_reset_auto / tt-smi -r can fail (exit 1) when the device is still
        claimed by a just-killed-but-not-yet-reaped hang, or is transiently
        busy. Free the device first and retry a few times with a backoff so it
        has time to be released, instead of giving up after a single immediate
        retry.
        """
        last_rc = None
        for attempt in range(1, self.reset_attempts + 1):
            try:
                self._free_device()
            except Exception as e:
                logger.warning(f"SWEEPS: device-holder cleanup failed (continuing to reset): {e}")

            # Surface tt-smi output on the final attempt to aid debugging.
            show_output = attempt == self.reset_attempts
            result = subprocess.run(
                [self.command, *self.args],
                stdout=None if show_output else subprocess.DEVNULL,
            )
            last_rc = result.returncode
            if last_rc == 0:
                logger.info(f"TT-SMI Reset Complete Successfully (attempt {attempt}/{self.reset_attempts})")
                return
            if attempt < self.reset_attempts:
                logger.warning(
                    f"SWEEPS: TT-SMI reset attempt {attempt}/{self.reset_attempts} failed (exit {last_rc}); "
                    f"waiting {self.reset_backoff_seconds}s for the device to be released, then retrying."
                )
                time.sleep(self.reset_backoff_seconds)

        raise RuntimeError(
            f"SWEEPS: TT-SMI Reset Failed with Exit Code: {last_rc} after {self.reset_attempts} attempts"
        )
