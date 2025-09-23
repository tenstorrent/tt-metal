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
    generate_logs_folder,
)


def profile_command(test_command, output_folder, name_append, collect_noc_traces):
    currentEnvs = dict(os.environ)
    currentEnvs["TT_METAL_DEVICE_PROFILER"] = "1"

    options = ""
    if output_folder:
        options += f"-o {output_folder}"
    if name_append:
        options += f" -n {name_append}"
    if collect_noc_traces:
        options += f" --collect-noc-traces "

    opProfilerTestCommand = f"python3 -m tracy -v -r -p {options} -m '{test_command}'"
    subprocess.run([opProfilerTestCommand], shell=True, check=False, env=currentEnvs)


@click.command()
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for artifacts")
@click.option("-c", "--command", type=str, required=True, help="Test command to profile")
@click.option("-n", "--name-append", type=str, help="Name to be appended to artifact names and folders")
@click.option("--collect-noc-traces", is_flag=True, default=False)
def main(command, output_folder, name_append, collect_noc_traces):
    logger.warning(
        "profile_this.py is getting deprecated soon. Please use the tracy.py module with -r option to obtain op reports."
    )
    if command:
        logger.info(f"profile_this.py is running {command}")
        profile_command(command, output_folder, name_append, collect_noc_traces)


if __name__ == "__main__":
    main()
