#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import sys
import argparse
from collections import defaultdict
import tempfile

DEBUG = False
TT_METAL_HOME = os.environ["TT_METAL_HOME"]
DEFAULT_GTEST = f"{TT_METAL_HOME}/build/test/tt_metal/unit_tests"


def extract_list_of_test_points(args: argparse.Namespace):
    p = subprocess.run(
        f"{args.gtest} --gtest_list_tests",
        stdout=subprocess.PIPE,
        shell=True,
    )

    data = defaultdict(lambda: list())
    key = None
    for line in p.stdout.decode("utf-8").split("\n"):
        if len(line.strip()) < 1:
            continue
        if line.endswith("."):
            key = line.strip()[:-1]
            data[key] = list()
            continue
        elif line.startswith(" "):
            data[key].append(line.split("#")[0].strip())
        else:
            print("SKIP line %s" % (line))
    return data


def run_testpoints_per_process(args: argparse.Namespace, data: dict):
    boilerplate = """
    #/bin/bash

    set -eo pipefail

    if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
    fi

    cd $TT_METAL_HOME

    export PYTHONPATH=$TT_METAL_HOME

    """

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "test.sh")
        with open(fname, "w") as fp:
            fp.write(boilerplate)
            total = 0
            for k, values in data.items():
                total += len(values)
            count = 0
            for k, values in data.items():
                for v in values:
                    fp.write(f"echo Running unittest {count}/{total}\n")
                    fp.write(f"{args.gtest} --gtest_filter='{k}.{v}'\n")
                    # fp.write(f"sleep 2\n")
                    count += 1
        if DEBUG:
            subprocess.run("cat " + fname, shell=True)
        p = subprocess.run("/bin/bash " + fname, shell=True)
        sys.exit(p.returncode)


def get_args():
    parser = argparse.ArgumentParser(
        usage="run_testpoint_perprocess.py --gtest /path/to/gtest/",
        description=f"run gtest with each test point in test suite as an independent process; required to have TT_METAL_HOME environment variable setup. Defaults to {DEFAULT_GTEST}",
    )
    parser.add_argument(
        "--gtest",
        help="specify full path to gtest to be run as independent process per test point in suite",
        required=False,
        default=DEFAULT_GTEST,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data = extract_list_of_test_points(args)
    run_testpoints_per_process(args, data)
