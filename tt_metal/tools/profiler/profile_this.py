#! /usr/bin/env python3

import os
import subprocess

import click
from loguru import logger

ENVS = dict(os.environ)
TT_METAL_HOME = ENVS["TT_METAL_HOME"]
LOG_LOCATIONS_RECORD = "tt_metal/tools/profiler/logs/.locations.log"

def test_profiler_build():
    from libs import tt_lib as ttl

    tmpLocation ="tt_metal/tools/profiler/tmp"
    os.system(f"rm -rf {tmpLocation}")
    ret = False

    ttl.profiler.set_profiler_location(tmpLocation)
    ttl.profiler.start_profiling("test")
    ttl.profiler.stop_profiling("test")

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


def get_log_locations ():
    logLocations = []
    with open(LOG_LOCATIONS_RECORD, "r") as recordFile:
        for line in recordFile.readlines():
            logLocation = line.strip()
            if os.path.isdir(f"{TT_METAL_HOME}/{logLocation}"):
                logLocations.append(logLocation)
    return logLocations

def post_process():
    logLocations = get_log_locations()
    for logLocation in logLocations:
        testName = logLocation.split("/")[-1]
        if testName == "ops":
            testName = "default"
        outputLocation = f"tt_metal/tools/profiler/output/ops/{testName}"
        os.system(f"tt_metal/tools/profiler/process_ops_logs.py -i {logLocation} -o {outputLocation}")
        logger.info(f"Post processed {testName} with results save in {outputLocation}")


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
