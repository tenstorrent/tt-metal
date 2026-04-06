# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import time
import subprocess
import click
from glob import glob
from loguru import logger


@click.command()
@click.option("-d", "--directory", required=True, type=str, help="Directory with test files")
@click.option("-r", "--result", default="result_sweeps", type=str, help="Directory to save results")
def main(directory, result):
    """
    Example for running sweep tests:

    python tests/tt_eager/python_api_testing/sweep_tests/run_sweep_tests.py -d tests/tt_eager/python_api_testing/sweep_tests/test_configs/ci_sweep_tests_working/grayskull/ -r result_sweeps
    """

    yaml_files = glob(os.path.join(directory, "*.yaml"))
    yaml_files.sort()
    skip = False

    for yaml_file in yaml_files:
        basename = os.path.splitext(os.path.basename(yaml_file))[0]

        if basename == "ttnn_eltwise_sub_test":
            skip = False

        if skip:
            continue

        outFolder = f"{result}/{basename}"
        command = f"pytest tests/tt_eager/python_api_testing/sweep_tests/run_sweep_test.py --input-path {yaml_file} --input-method cli --cli-input {outFolder}"

        start = time.time()
        subprocess.run([command], shell=True, check=False)
        duration = time.time() - start

        try:
            with open(f"{outFolder}/total_time.txt", "w") as file:
                file.write(f"{duration:.2f}")
        except:
            logger.error(f"Could not write total_time to total_time.txt probably sweep {yaml_file} crashed")


if __name__ == "__main__":
    main()
