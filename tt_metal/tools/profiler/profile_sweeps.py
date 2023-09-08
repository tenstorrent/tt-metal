"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import os
import time
import subprocess
from glob import glob
from loguru import logger
from tt_metal.tools.profiler.profile_this import (
    test_profiler_build,
    profile_command,
    get_log_locations,
)


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


directory = "tests/tt_eager/python_api_testing/sweep_tests/test_configs/ci_sweep_tests"
result_folder = "/home/ubuntu/tt-metal/ng-test-sweeps/"

if __name__ == "__main__":
    if test_profiler_build():
        logger.info(f"Profiler build flag is set")
    else:
        assert (
            False
        ), "Need to build with the profiler flag enabled. i.e. make build ENABLE_PROFILER=1"

    txt_files = glob(os.path.join(directory, "*.yaml"))
    txt_files.sort()
    do_run = True

    for txt_file in txt_files:
        basename = os.path.splitext(os.path.basename(txt_file))[0]
        command = f"python tests/tt_eager/python_api_testing/sweep_tests/run_pytorch_test.py -i {txt_file} -o {result_folder}{basename}"
        profile_output_folder = f"{result_folder}{basename}/profile"

        # if basename == "pytorch_add_layernorm_test":
        #     do_run = True

        # if basename == "pytorch_eltwise_asinh_test":
        #     break

        if do_run:
            print(command)

            subprocess.run(
                ["rm -rf tt_metal/tools/profiler/logs/ops_device"],
                shell=True,
                check=True,
            )
            subprocess.run(
                ["rm -rf tt_metal/tools/profiler/logs/ops"], shell=True, check=True
            )

            start = time.time()
            profile_command(command)
            duration = time.time() - start

            post_process(f"{result_folder}{basename}/profile")

            with open(f"{result_folder}{basename}/total_time.txt", "w") as file:
                file.write(f"{duration:.2f}")

            # break
