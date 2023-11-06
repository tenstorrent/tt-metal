#! /usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess

import click
from loguru import logger

from tt_metal.tools.profiler.common import (
    PROFILER_SCRIPTS_ROOT,
    TT_METAL_HOME,
    PROFILER_OUTPUT_DIR,
    get_log_locations,
    test_profiler_build,
)


def profile_command(test_command, device_only=False, host_only=False):
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


def post_process(outputLocation=None, nameAppend=""):
    logLocations = get_log_locations()

    for logLocation in logLocations:
        if outputLocation is None:
            outputLocation = PROFILER_OUTPUT_DIR

        postProcessCmd = f"{PROFILER_SCRIPTS_ROOT}/process_ops_logs.py -i {logLocation} -o {outputLocation} --date"
        if nameAppend:
            postProcessCmd += f" -n {nameAppend}"

        os.system(postProcessCmd)


@click.command()
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for artifacts")
@click.option("-c", "--command", type=str, required=True, help="Test command to profile")
@click.option("-n", "--name-append", type=str, help="Name to be appended to artifact names and folders")
@click.option(
    "-d",
    "--device-only",
    default=False,
    is_flag=True,
    help="Only do device side profiling, note in this mode host side readings will still be reported but should be ignored",
)
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
