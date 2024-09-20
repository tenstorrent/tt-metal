# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess

GRAYSKULL_ARGS = ["-tr", "all"]
WORMHOLE_ARGS = ["-wr", "all"]

RESET_OVERRIDE = os.getenv("TT_SMI_RESET_COMMAND")


def run_tt_smi(arch: str):
    if RESET_OVERRIDE is not None:
        smi_process = subprocess.run(RESET_OVERRIDE, shell=True)
        if smi_process.returncode == 0:
            print("SWEEPS: TT-SMI Reset Complete Successfully")
            return
        else:
            raise Exception(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {smi_process.returncode}")
        return

    if arch not in ["grayskull", "wormhole_b0"]:
        raise Exception(f"SWEEPS: Unsupported Architecture for TT-SMI Reset: {arch}")

    smi_options = [
        "tt-smi",
        "tt-smi-metal",
        "/home/software/syseng/gs/tt-smi" if arch == "grayskull" else "/home/software/syseng/wh/tt-smi",
    ]
    args = GRAYSKULL_ARGS if arch == "grayskull" else WORMHOLE_ARGS

    # Potential implementation to use the pre-defined reset script on each CI runner. Not in use because of the time and extra functions it has. (hugepages check, etc.)
    # if os.getenv("CI") == "true":
    #     smi_process = subprocess.run(f"/opt/tt_metal_infra/scripts/ci/{arch}/reset.sh", shell=True, capture_output=True)
    #     if smi_process.returncode == 0:
    #         print("SWEEPS: TT-SMI Reset Complete Successfully")
    #         return
    #     else:
    #         raise Exception(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {smi_process.returncode}")

    for smi_option in smi_options:
        executable = shutil.which(smi_option)
        if executable is not None:
            # Corner case for newer version of tt-smi, -tr and -wr are removed on this version (tt-smi-metal). Default device 0, if needed use TT_SMI_RESET_COMMAND override.
            if smi_option == "tt-smi-metal":
                args = ["-r", "0"]
            smi_process = subprocess.run([executable, *args])
            if smi_process.returncode == 0:
                print("SWEEPS: TT-SMI Reset Complete Successfully")
                return
            else:
                raise Exception(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {smi_process.returncode}")

    raise Exception("SWEEPS: Unable to Locate TT-SMI Executable")
