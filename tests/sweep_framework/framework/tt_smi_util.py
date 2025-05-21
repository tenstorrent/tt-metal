# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger

GRAYSKULL_ARGS = ["-tr", "all"]
LEGACY_WORMHOLE_ARGS = ["-wr", "all"]


class ResetUtil:
    def __init__(self, arch: str):
        self.arch = arch
        self.command = os.getenv("TT_SMI_RESET_COMMAND")
        self.args = []
        if arch not in ["grayskull", "wormhole_b0", "blackhole"]:
            raise Exception(f"SWEEPS: Unsupported Architecture for TT-SMI Reset: {arch}")
        if self.command is not None:
            return

        self.smi_options = [
            "tt-smi",
            "tt-smi-metal",
            "/home/software/syseng/gs/tt-smi" if arch == "grayskull" else "/home/software/syseng/wh/tt-smi",
        ]
        for smi_option in self.smi_options:
            executable = shutil.which(smi_option)
            if executable is not None:
                args = []
                # Corner case for newer version of tt-smi, -tr and -wr are removed on this version (tt-smi-metal).
                # Default device 0, if needed use TT_SMI_RESET_COMMAND override.
                if smi_option == "tt-smi-metal":
                    args = ["-r", "0"]
                elif arch == "grayskull":
                    args = GRAYSKULL_ARGS
                elif arch == "wormhole_b0":
                    smi_process = subprocess.run([executable, "-v"], capture_output=True, text=True)
                    smi_version = smi_process.stdout.strip()
                    if not smi_version.startswith("3.0"):
                        args = LEGACY_WORMHOLE_ARGS
                    else:
                        args = ["-r"]
                        smi_process = subprocess.run([executable, "-g", ".reset.json"])
                        import json

                        with open(".reset.json", "r") as f:
                            reset_json = json.load(f)
                            card_id = reset_json["wh_link_reset"]["pci_index"]
                            if len(card_id) < 1:
                                raise Exception(f"SWEEPS: TT-SMI Reset Failed to Find Card ID.")
                            args.append(str(card_id[0]))
                        subprocess.run(["rm", "-f", ".reset.json"])
                elif arch == "blackhole":
                    args = ["-r", "0"]
                smi_process = subprocess.run([executable, *args])
                if smi_process.returncode == 0:
                    self.command = executable
                    self.args = args
                    break

        if self.command is None:
            raise Exception(f"SWEEPS: Unable to location tt-smi executable")
        print(f"SWEEPS: tt-smi util initialized with command: {self.command}, args: {self.args}")

    def reset(self):
        smi_process = subprocess.run([self.command, *self.args], stdout=subprocess.DEVNULL)
        if smi_process.returncode == 0:
            logger.info("TT-SMI Reset Complete Successfully")
            return
        else:
            raise Exception(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {smi_process.returncode}")

    # Potential implementation to use the pre-defined reset script on each CI runner. Not in use because of the time and extra functions it has. (hugepages check, etc.)
    # if os.getenv("CI") == "true":
    #     smi_process = subprocess.run(f"/opt/tt_metal_infra/scripts/ci/{arch}/reset.sh", shell=True, capture_output=True)
    #     if smi_process.returncode == 0:
    #         print("SWEEPS: TT-SMI Reset Complete Successfully")
    #         return
    #     else:
    #         raise Exception(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {smi_process.returncode}")
