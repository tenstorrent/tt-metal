# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from multiprocessing import Process
from time import sleep
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger

LEGACY_WORMHOLE_ARGS = ["-wr", "all"]

DEVICE_PROBE_TIMEOUT = 30


def _device_probe_worker():
    """Open and close all available devices to verify Metal can initialize cleanly.

    Runs in a subprocess so that if the device open hangs (due to stale
    dispatch kernels from a previous module), the parent can kill it and
    trigger a tt-smi reset.
    """
    import ttnn

    num_devices = ttnn.get_num_devices()
    if num_devices > 1:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, num_devices),
            dispatch_core_config=ttnn.DispatchCoreConfig(),
        )
        ttnn.close_mesh_device(mesh_device)
        del mesh_device
    else:
        device = ttnn.open_device(device_id=0)
        ttnn.close_device(device)


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
        """Quick sanity check: open and close a device in a subprocess.

        If the probe completes within the timeout, devices are clean.
        If it hangs or crashes (stale dispatch kernels, dirty state),
        a full tt-smi reset is performed automatically.

        Returns True if devices were already healthy, False if a reset
        was needed.
        """
        probe = Process(target=_device_probe_worker, daemon=True)
        probe.start()
        probe.join(timeout=DEVICE_PROBE_TIMEOUT)

        if probe.is_alive():
            logger.warning(
                f"Device health probe hung (>{DEVICE_PROBE_TIMEOUT}s) — stale device state detected. "
                "Killing probe and performing tt-smi reset."
            )
            probe.kill()
            probe.join()
            self.reset()
            return False

        if probe.exitcode != 0:
            logger.warning(f"Device health probe failed (exit code {probe.exitcode}). " "Performing tt-smi reset.")
            self.reset()
            return False

        logger.info("Device health probe passed — devices are clean.")
        return True
