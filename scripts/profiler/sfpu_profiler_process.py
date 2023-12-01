#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import sys
import os
import subprocess
import copy
from pprint import pprint
import argparse

# ASSUME: repo is compiled for the profiling
# ASSUME: test_sfpu is compiled
results = {}


def make_parser():
    parser = argparse.ArgumentParser(prog="sfpu_profiler_process.py", description="run SFPU profiler")
    all_operations = [
        "abs",
        "exponential",
        "gelu",
        "identity",
        "log",
        "log10",
        "log2",
        "reciprocal",
        "relu",
        "sigmoid",
        "sign",
        "sqrt",
        "square",
        "tanh",
    ]
    parser.add_argument("operations", nargs="+", default=all_operations, help=": " + " ".join(all_operations))
    parser.add_argument("--use-L1", action="store_true", default=False)
    return parser


def get_args():
    parser = make_parser()
    return parser.parse_args()


def run(args, home, function):
    testpath = os.path.join(home, "build", "test", "tt_eager", "ops", "test_sfpu")
    assert os.path.exists(testpath)
    env = copy.deepcopy(os.environ)
    env["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TT_METAL_ENV"] = "dev"
    testcmd = " ".join([testpath, function, args.use_L1 and "--use-L1" or "--use-DRAM", "--tile-factor", "1024"])
    subprocess.run(testcmd, shell=True, env=env)
    return


def postproc(home):
    def get_cycles(df):
        cycles = (
            df[df[" timer_id"] == 9998][" time[cycles since reset]"].to_numpy()
            - df[df[" timer_id"] == 9997][" time[cycles since reset]"].to_numpy()
        )[1]
        return cycles

    filepath = os.path.join(home, "generated", "profiler", ".logs", "profile_log_device.csv")
    df = pd.read_csv(filepath, header=1)
    print(f"the number of cycles (TRISC_1) = {get_cycles(df)}")
    return get_cycles(df)


def main():
    args = get_args()

    def run_function(function):
        home = os.environ["TT_METAL_HOME"]
        run(args, home, function)
        cycles = postproc(home)
        results[function] = cycles

    list(map(run_function, args.operations))
    pprint(["*" * 40, "result", "*" * 40])
    print(pd.DataFrame.from_dict(results, orient="index", columns=["TRISC_1 Cycles"]))
    print("*" * 80)


if __name__ == "__main__":
    main()
