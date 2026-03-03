# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger

LEGACY_WORMHOLE_ARGS = ["-wr", "all"]


class ResetUtil:
    SUPPORTED_ARCHS = {"wormhole_b0", "blackhole"}

    def __init__(self, arch: str):
        if arch not in self.SUPPORTED_ARCHS:
            raise ValueError(f"SWEEPS: Unsupported Architecture for TT-SMI Reset: {arch}")

        self.arch = arch
        self.command, self.args = self._find_command()

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

    def reset(self):
        """Execute the reset command."""
        result = subprocess.run([self.command, *self.args], stdout=subprocess.DEVNULL)
        if result.returncode != 0:
            # give it one more try
            result = subprocess.run([self.command, *self.args])
            if result.returncode != 0:
                raise RuntimeError(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {result.returncode}")
        logger.info("TT-SMI Reset Complete Successfully")

    def ensure_device_health(self):
        """Open and close all devices in an isolated subprocess, capturing stderr.

        Detects three failure modes:
        1. Probe hangs (>timeout)     -> stale dispatch kernels blocking init
        2. Probe crashes (non-zero)   -> fatal device error
        3. Probe succeeds BUT stderr contains stale-state warnings
           (e.g. "still running", "unexpected run_mailbox")
           -> Metal's internal recovery masked the problem, but devices
              are not truly clean

        Returns True if devices are healthy, False if a reset was needed.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-c", _DEVICE_PROBE_SCRIPT],
                timeout=DEVICE_PROBE_TIMEOUT,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Device health probe hung (>{DEVICE_PROBE_TIMEOUT}s) — stale device state detected. "
                "Performing tt-smi reset."
            )
            self.reset()
            return False

        combined_output = (result.stdout or "") + (result.stderr or "")

        if result.returncode != 0:
            logger.warning(f"Device health probe crashed (exit code {result.returncode}). Performing tt-smi reset.")
            if combined_output:
                for line in combined_output.strip().splitlines()[-5:]:
                    logger.warning(f"  probe: {line.strip()}")
            self.reset()
            return False

        output_lower = combined_output.lower()
        for marker in _STALE_DEVICE_MARKERS:
            if marker.lower() in output_lower:
                logger.warning(
                    f"Device health probe detected stale state ('{marker}' in output). " "Performing tt-smi reset."
                )
                self.reset()
                return False

        logger.info("Device health probe passed — devices are clean.")
        return True
