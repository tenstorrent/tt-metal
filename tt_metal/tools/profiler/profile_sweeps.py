# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import os
import sys
import time
import subprocess
import click
from glob import glob
from loguru import logger
from tt_metal.tools.profiler.profile_this import (
    test_profiler_build,
    profile_command,
)


ENVS = dict(os.environ)
TT_METAL_HOME = ENVS["TT_METAL_HOME"]
LOG_LOCATIONS_RECORD = "tt_metal/tools/profiler/logs/.locations.log"


def get_log_locations ():
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

        os.system(f"python tt_metal/tools/profiler/process_ops_logs.py -i {logLocation} -o {outputLocation}")
        logger.info(f"Post processed {testName} with results saved in {outputLocation}")


@click.command()
@click.option("-d", "--directory", type=str, help="Directory with test files")
@click.option("-r", "--result", type=str, help="Directory to save results")
def main(directory, result):
    if test_profiler_build():
        logger.info(f"Profiler build flag is set")
    else:
        assert False, "Need to build with the profiler flag enabled. i.e. make build ENABLE_PROFILER=1"

    txt_files = glob(os.path.join(directory, "*.yaml"))
    txt_files.sort()
    do_run = True

    for txt_file in txt_files:
        basename = os.path.splitext(os.path.basename(txt_file))[0]
        command = f"python tests/tt_eager/python_api_testing/sweep_tests/run_pytorch_test.py -i {txt_file} -o {result}{basename}"
        profile_output_folder = f"{result}{basename}/profile"

        if do_run:
            print(command)

            subprocess.run(
                ["rm -rf tt_metal/tools/profiler/logs/ops_device"],
                shell=True,
                check=True,
            )
            subprocess.run(["rm -rf tt_metal/tools/profiler/logs/ops"], shell=True, check=True)

            start = time.time()
            profile_command(command)
            duration = time.time() - start

            post_process(f"{result}{basename}/profile")

            with open(f"{result}{basename}/total_time.txt", "w") as file:
                file.write(f"{duration:.2f}")


if __name__ == "__main__":
    main()
