# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser(prog="BUDA-Eager kernels prepare")
    parser.add_argument("action", choices=["prepare", "clean"])
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()

    short = args.short
    action = args.action

    tt_metal_home_str = os.environ.get("TT_METAL_HOME", None)

    if not tt_metal_home_str:
        raise Exception("TT_METAL_HOME must be set before you run this script")

    set_up_kernels_cmd = (
        f"make -f {tt_metal_home_str}/tt_metal/hw/Makefile-runtime -C {tt_metal_home_str}/tt_metal/hw {action}".split(
            " "
        )
    )

    if short:
        pass
    else:
        print(set_up_kernels_cmd)

    subprocess.check_call(set_up_kernels_cmd, env=os.environ.copy())
