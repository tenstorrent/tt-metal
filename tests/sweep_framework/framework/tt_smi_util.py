# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from time import sleep
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger


class ResetUtil:
    def __init__(self, arch: str):
        self.arch = arch
        self.command = os.getenv("TT_SMI_RESET_COMMAND")
        self.args = []
        if arch not in ["wormhole_b0", "blackhole"]:
            raise Exception(f"SWEEPS: Unsupported Architecture for TT-SMI Reset: {arch}")
        if self.command is not None:
            command_parts = self.command.split()
            self.command = command_parts[0]
            self.args = command_parts[1:]
            return

        self.smi_options = ["tt-smi"]
        for smi_option in self.smi_options:
            executable = shutil.which(smi_option)
            logger.info(f"tt-smi executable: {executable}")
            if executable is not None:
                args = []
                # Corner case for newer version of tt-smi, -tr and -wr are removed on this version (tt-smi-metal).
                # Default device 0, if needed use TT_SMI_RESET_COMMAND override.
                if smi_option == "tt-smi-metal":
                    args = ["-r", "0"]
                elif arch == "wormhole_b0":
                    smi_process = subprocess.run([executable, "-v"], capture_output=True, text=True)
                    smi_version = smi_process.stdout.strip()
                    if not smi_version.startswith("3.0"):
                        args = LEGACY_WORMHOLE_ARGS
                    else:
                        args = ["-r"]

                self.command = executable
                self.args = args

                self.reset()
                return

        if self.command is None:
            raise Exception(f"SWEEPS: Unable to locate tt-smi executable")
            
        self.reset()
        return
        

    def reset(self):
        smi_process = subprocess.run([self.command, *self.args], stdout=subprocess.DEVNULL)
        if smi_process.returncode == 0:
            logger.info("TT-SMI Reset Complete Successfully")
            return
        else:
            raise Exception(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {smi_process.returncode}")
