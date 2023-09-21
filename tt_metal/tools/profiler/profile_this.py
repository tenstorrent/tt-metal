#! /usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from tt_eager import tt_lib
import click
from loguru import logger

ENVS = dict(os.environ)
TT_METAL_HOME = ENVS["TT_METAL_HOME"]
LOG_LOCATIONS_RECORD = "tt_metal/tools/profiler/logs/.locations.log"


def test_profiler_build():
    tmpLocation = "tt_metal/tools/profiler/tmp"
    os.system(f"rm -rf {tmpLocation}")
    ret = False

    tt_lib.profiler.set_profiler_location(tmpLocation)
    tt_lib.profiler.start_profiling("test")
    tt_lib.profiler.stop_profiling("test")

    if os.path.isfile(f"{tmpLocation}/test/profile_log_host.csv"):
        ret = True

    os.system(f"rm -rf {tmpLocation}")
    return ret


def profile_command(test_command):
    currentEnvs = dict(os.environ)
    currentEnvs["TT_METAL_DEVICE_PROFILER"] = "1"
    currentEnvs["TT_PCI_DMA_BUF_SIZE"] = "1048576"
    subprocess.run([test_command], shell=True, check=True, env=currentEnvs)

    currentEnvs = dict(os.environ)
    currentEnvs["TT_METAL_DEVICE_PROFILER"] = "0"
    subprocess.run([test_command], shell=True, check=True, env=currentEnvs)


def get_log_locations():
    logLocations = []
    with open(LOG_LOCATIONS_RECORD, "r") as recordFile:
        for line in recordFile.readlines():
            logLocation = line.strip()
            if os.path.isdir(f"{TT_METAL_HOME}/{logLocation}"):
                logLocations.append(logLocation)
    return logLocations


def post_process(outputLocation=None):
    logLocations = get_log_locations()

    for logLocation in logLocations:
        testName = logLocation.split("/")[-1]
        if testName == "ops":
            testName = "default"

        if outputLocation is None:
            outputLocation = f"tt_metal/tools/profiler/output/ops/{testName}"

        os.system(f"tt_metal/tools/profiler/process_ops_logs.py -i {logLocation} -o {outputLocation}")
        logger.info(f"Post processed {testName} with results saved in {outputLocation}")


@click.command()
@click.option("-c", "--command", type=str, help="Test command to profile")
def main(command):
    isProfilerBuild = test_profiler_build()
    if isProfilerBuild:
        logger.info(f"Profiler build flag is set")
    else:
        assert False, "Need to build with the profiler flag enabled. i.e. make build ENABLE_PROFILER=1"

    if command:
        profile_command(command)
        post_process()


if __name__ == "__main__":
    main()
