#! /usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess

import click
from loguru import logger

ENVS = dict(os.environ)
TT_METAL_HOME = ""
if "TT_METAL_HOME" in ENVS.keys():
    TT_METAL_HOME = ENVS["TT_METAL_HOME"]
else:
    logger.error("TT_METAL_HOME environment variable is not set up properly.")
    sys.exit(1)

PROFILER_ROOT = TT_METAL_HOME + "/tt_metal/tools/profiler/"
LOG_LOCATIONS_RECORD = PROFILER_ROOT + "logs/.locations.log"


def test_profiler_build():
    from tt_eager import tt_lib

    tmpLocation = f"{PROFILER_ROOT}/tmp"
    os.system(f"rm -rf {tmpLocation}")
    ret = False

    tt_lib.profiler.set_profiler_location(tmpLocation)
    tt_lib.profiler.start_profiling("test")
    tt_lib.profiler.stop_profiling("test")

    if os.path.isfile(f"{tmpLocation}/test/profile_log_host.csv"):
        ret = True
    elif os.path.isfile(f"{tmpLocation}_device/test/profile_log_host.csv"):
        ret = True

    os.system(f"rm -rf {tmpLocation}")
    return ret


def profile_command(test_command, device_only = False, host_only = False):
    doBoth = True
    if device_only or host_only:
        doBoth = False

    if doBoth or device_only:
        currentEnvs = dict(os.environ)
        currentEnvs["TT_METAL_DEVICE_PROFILER"] = "1"
        subprocess.run([test_command], shell=True, check=False, env=currentEnvs)

    if doBoth or host_only:
        currentEnvs = dict(os.environ)
        currentEnvs["TT_METAL_DEVICE_PROFILER"] = "0"
        subprocess.run([test_command], shell=True, check=False, env=currentEnvs)


def get_log_locations():
    logLocations = set()
    deviceLogLocations = set()
    if os.path.isfile(LOG_LOCATIONS_RECORD):
        with open(LOG_LOCATIONS_RECORD, "r") as recordFile:
            for line in recordFile.readlines():
                logLocation = line.strip()
                if os.path.isdir(f"{TT_METAL_HOME}/{logLocation}"):
                    logLocations.add(logLocation)
                    tmpSplit = logLocation.rsplit("_",1)
                    if  tmpSplit[-1] == "device":
                        deviceLogLocations.add(tmpSplit[0])
    for logLocation in deviceLogLocations:
        if logLocation in logLocations:
            logLocations.remove(f"{logLocation}_device")

    return list(logLocations)


def post_process(outputLocation=None, nameAppend=""):
    logLocations = get_log_locations()

    for logLocation in logLocations:
        if outputLocation is None:
            outputLocation = os.path.join(TT_METAL_HOME, ".profiler")

        postProcessCmd = f"{PROFILER_ROOT}/process_ops_logs.py -i {logLocation} -o {outputLocation} --date"
        if nameAppend:
            postProcessCmd += f" -n {nameAppend}"

        os.system(postProcessCmd)


@click.command()
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for artifacts")
@click.option("-c", "--command", type=str, required=True, help="Test command to profile")
@click.option("-n", "--name-append", type=str, help="Name to be appended to artifact names and folders")
@click.option("-d", "--device-only", default=False, is_flag=True, help="Only do device side profiling, note in this mode host side readings will still be reported but should be ignored")
@click.option("-m", "--host-only", default=False, is_flag=True, help="Only do host side profiling")
def main(command, output_folder, name_append, device_only, host_only):
    isProfilerBuild = test_profiler_build()
    if isProfilerBuild:
        logger.info(f"Profiler build flag is set")
    else:
        logger.error(f"Need to build with the profiler flag enabled. i.e. make build ENABLE_PROFILER=1")
        sys.exit(1)

    if command:
        profile_command(command, device_only, host_only)
        post_process(output_folder, name_append)


if __name__ == "__main__":
    main()
