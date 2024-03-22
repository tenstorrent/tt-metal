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
    txt_files = glob(os.path.join(directory, "*.yaml"))
    txt_files.sort()

    for txt_file in txt_files:
        basename = os.path.splitext(os.path.basename(txt_file))[0]
        outFolder = f"{result}/{basename}"
        command = (
            f"python tests/tt_eager/python_api_testing/sweep_tests/run_pytorch_test.py -i {txt_file} -o {outFolder}"
        )

        start = time.time()
        subprocess.run([command], shell=True, check=False)
        duration = time.time() - start

        with open(f"{outFolder}/total_time.txt", "w") as file:
            file.write(f"{duration:.2f}")


if __name__ == "__main__":
    main()
