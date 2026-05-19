# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger


def _is_galaxy_6u():
    """Detect if running on a Galaxy 6U topology based on runner label env var."""
    runner_label = os.getenv("RUNNER_LABEL", "")
    tt_smi_cmd = os.getenv("TT_SMI_RESET_COMMAND", "")
    return "topology-6u" in runner_label or "glx_reset" in tt_smi_cmd


def _perform_eth_retrain():
    """Run UMD topology discovery with ETH link retraining for 6U machines.

    This ensures Ethernet links are fully trained after reset, which tt-smi
    alone does not guarantee on WH 6U (Galaxy) topologies.
    """
    try:
        import tt_umd

        logger.info("SWEEPS: Performing 6U ETH link retrain via UMD TopologyDiscovery...")
        options = tt_umd.TopologyDiscoveryOptions()
        options.perform_6u_eth_retrain = True
        options.eth_fw_heartbeat_failure = tt_umd.TopologyDiscoveryOptions.Action.IGNORE
        tt_umd.TopologyDiscovery.discover(options)
        logger.info("SWEEPS: 6U ETH retrain completed successfully")
    except ImportError:
        logger.warning("SWEEPS: tt_umd not available, skipping ETH retrain")
    except Exception as e:
        logger.warning(f"SWEEPS: ETH retrain failed (non-fatal): {e}")


class ResetUtil:
    SUPPORTED_ARCHS = {"wormhole_b0", "blackhole"}

    def __init__(self, arch: str):
        if arch not in self.SUPPORTED_ARCHS:
            raise ValueError(f"SWEEPS: Unsupported Architecture for TT-SMI Reset: {arch}")

        self.arch = arch
        self.command, self.args = self._find_command()
        self._is_6u = _is_galaxy_6u()

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
        """Execute the reset command, followed by ETH retrain on 6U."""
        result = subprocess.run([self.command, *self.args], stdout=subprocess.DEVNULL)
        if result.returncode != 0:
            # give it one more try
            result = subprocess.run([self.command, *self.args])
            if result.returncode != 0:
                raise RuntimeError(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {result.returncode}")
        logger.info("TT-SMI Reset Complete Successfully")

        if self._is_6u and self.arch == "wormhole_b0":
            _perform_eth_retrain()
