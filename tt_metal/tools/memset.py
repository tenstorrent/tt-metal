# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
from subprocess import Popen, PIPE
import argparse
import os
import sys

if __name__ == "__main__":
    assert "TT_METAL_HOME" in os.environ, "TT_METAL_HOME must be set"
    TT_METAL_HOME = os.environ["TT_METAL_HOME"]
    executable = f"{TT_METAL_HOME}/build/obj/tt_metal/tools/memset"
    assert os.path.exists(executable), "You must compile the project first"

    command = executable.split(" ")

    parser = argparse.ArgumentParser(
        prog="memset.py",
        description="Launches a program that blasts either chip DRAM"
        "or L1 with a user-specified value for a certain range",
    )

    parser.add_argument("--mem_type", required=True, type=str, help="Either dram or l1")
    parser.add_argument("--chip_id", required=True, type=int, help="Which chip you want to target")
    parser.add_argument("--start_addr", required=True, type=int, help="Start address you want to target")
    parser.add_argument("--size", required=True, type=int, help="Size of the vector you want to write")
    parser.add_argument("--val", required=True, type=int, help="What value you want to write")
    args = parser.parse_args()

    # All correctness checks done here
    assert args.mem_type in ["l1", "dram"], "'mem_type' must be one of dram or l1"
    assert args.chip_id >= 0, "Cannot have a negative chip id"
    assert args.start_addr >= 0, "Cannot have a negative start address"
    assert args.size >= 0, "Cannot have a negative size"
    assert args.val >= 0, "Cannot write a negative value"

    command.extend([args.mem_type, args.chip_id, args.start_addr, args.size, args.val])

    # We kept the types before so that argparse can do type checking, but
    # we need to convert to string here so that we can pass the command
    # to process popen
    command = [str(entry) for entry in command]

    print(f"Running command: {command}\n")

    os.environ["RUNNING_FROM_PYTHON"] = "1"
    process = Popen(command, stderr=sys.stderr, stdout=sys.stdout)
    stdout, stderr = process.communicate()
