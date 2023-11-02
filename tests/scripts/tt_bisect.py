#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from collections import OrderedDict, namedtuple
import time
import subprocess
import json
import pathlib
import copy

ARCH_NAME = os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower()
HOME = pathlib.Path(os.path.abspath(__file__)).parent


def is_wormhole_b0():
    return "wormhole_b0" in ARCH_NAME


def is_grayskull():
    return "grayskull" in ARCH_NAME


def get_environment() -> dict:
    def make_env():
        env = copy.copy(os.environ)
        env.update(
            {
                "TT_METAL_ENV": "dev",
                "TT_METAL_HOME": os.environ["TT_METAL_HOME"],
                "PYTHONPATH": os.environ["TT_METAL_HOME"] + ":" + os.environ.get("PYTHONPATH", ""),
                # "ARCH_NAME": "",
            }
        )
        return env

    def make_gs():
        d = {"ARCH_NAME": "grayskull"}
        base_e = make_env()
        base_e.update(d)
        return base_e

    def make_wh_b0():
        d = {
            "TT_METAL_SLOW_DISPATCH_MODE": "1",
            "ARCH_NAME": "wormhole_b0",
        }
        base_e = make_env()
        base_e.update(d)
        return base_e

    if is_grayskull():
        return make_gs()
    assert is_wormhole_b0()
    return make_wh_b0()


class BashScript:
    while_loop_fmt = """
    i=1
    while [ $i -gt 0 ];
    do
        echo "Running command ${0} for {1}/{2} iteration"
        {0}  {1}
        i=$(( $i + 1 ))
    s
    done
    """

    start_sh_fmt = """
    export ARCH_NAME=wormhole_b0
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_ENV=dev
    export TT_METAL_HOME=/home/tt-admin/muthu-tt-metal
    export OLDPWD=/home/tt-admin/muthu-tt-metal/build/test/tt_metal
    source ./build/python_env/bin/activate
    cd ${TT_METAL_HOME}
    export PYTHONPATH=${TT_METAL_HOME}:${PYTHONPATH}
    (python_env) tt-admin@e09cs04:~/muthu-tt-metal$
    """

    def __init__(self, env):
        self.env = OrderedDict(env)

    @staticmethod
    def run_hugepages_check():
        p = subprocess.run(
            [
                "python3",
                f'{self.env["TT_METAL_HOME"]}/infra/machine_setup/scripts/setup_hugepages.py',
                "check",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=get_environment(),
        )
        return p.returncode == 0

    def export(self, filename):
        lines = []
        for var, value in self.env.items():
            lines.append(f"export {var}={value}")

        with open(filename, "w") as fp:
            fp.writelines(lines)
        return filename


class BisectArgs:
    def __init__(self):
        args = BisectArgs.make_parser().parse_args()
        self.good = args.good
        self.bad = args.bad
        self.test = args.test
        self.runs = args.runs
        self.arch = args.arch

    @property
    def testcmd(self):
        return self.get_test_cmd()

    def get_test_cmd(self):
        mapped = {
            "post-commit": "run_pre_post_commit_regressions.sh",
            "models": "run_models.sh",
            "performance": "run_performance.sh",
        }
        ref_path = os.path.join(os.environ["TT_METAL_HOME"], "tests", "scripts")
        return os.path.join(ref_path, mapped[self.test])

    @staticmethod
    def make_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("-g", "--good", type=str, default="HEAD")
        parser.add_argument("-b", "--bad", type=str, default="HEAD")
        parser.add_argument(
            "-t",
            "--test",
            choices=["post-commit", "models", "performance"],
            default="post-commit",
        )
        parser.add_argument("-r", "--runs", type=int, default=2)
        parser.add_argument(
            "-a",
            "--arch",
            choices=["grayskull", "wormhole_b0"],
            default=os.environ.get("ARCH_NAME", "grayskull"),
        )
        return parser


class Result:
    def __init__(self, tstart=None, tend=None, returncode=255, extras=None):
        self.tstart = tstart
        self.tend = tend
        self.returncode = returncode
        self.extras = extras

    def as_json(self):
        return {
            "tstart": self.tstart,
            "tend": self.tend,
            "returncode": self.returncode,
            "extras": self.extras,
        }


class StressTest:
    """prepare a repository (assumed) and run the tests @args.runs times and log results."""

    def __init__(self, args: BisectArgs):
        self.runs = args.runs
        self.testcmd = args.testcmd
        self.results = []  # list of Results
        self.filename = f"stress-test-log-{time.ctime()}.json"
        self._completed = False

    def __del__(self):
        if self._completed:
            self.save()
        return

    def save(self):
        with open(self.filename, "w") as fp:
            results = {}
            cmd_ip = {"testcmd": self.testcmd, "runs": self.runs}
            results.update(cmd_ip)
            results.update(results=self.results)
            fp.write(json.dumps(results, indent=True))
        return

    def run(self):
        for i in range(self.runs):
            start = time.ctime()
            p = None
            try:
                p = subprocess.run(
                    self.testcmd,
                    shell=True,
                    env=get_environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    executable="/bin/bash",
                )
            except Exception as _e:
                raise _e
            end = time.ctime()
            returncode = p and p.returncode or 255
            result = Result(
                start,
                end,
                returncode,
                p and {"output": str(p.stdout)[-2048:], "cmd": self.testcmd} or {},
            )
            if returncode != 0:
                print(
                    "RUN:",
                    i + 1,
                    f"/{self.runs} FAILED: ",
                    p and p.stdout.strip() or "p undefined",
                )
            else:
                print(
                    "RUN:",
                    i + 1,
                    f"/{self.runs} PASSED: ",
                    p and p.stdout.strip() or "p undefined",
                )

            self.results.append(result.as_json())
        self._completed = True


class GitBisect:
    def __int__(self):
        raise NotImplementedError()

    def __del__(self):
        self.run_cmd("reset")

    def run_cmd(self, cmd, *args):
        cmds = ["bisect", cmd, *args]

    def cmd_start(self):
        """
         $ git bisect start

        :return:
        """
        self.run_cmd("start")

    def cmd_bad(self, shaid):
        """
         $ git bisect bad HEAD
        :return:
        """
        self.run_cmd("bad", shaid)

    def cmd_good(self, shaid):
        """
         $ git bisect good c5b6863

        :return:
        """
        self.run_cmd("good", shaid)


def main():
    args = BisectArgs()
    if args.bad == args.good:
        StressTest(args).run()
        sys.exit(0)
    raise NotImplementedError()


if __name__ == "__main__":
    main()
