#! /usr/bin/env python3

import os, sys
import re

import click

from device_log_run import beautify_tt_js_blob

try:
    REPO_PATH = os.environ["PYTHONPATH"]
except KeyError:
    print("PYTHONPATH has to be setup. Please refer to getting started docs", file=sys.stderr)
    sys.exit(1)


TT_METAL_PATH = f"{REPO_PATH}/tt_metal"
GOLDEN_OUTPUTS_DIR = f"{TT_METAL_PATH}/third_party/lfs/profiler/tests/golden/device/outputs"
GOLDEN_LOGS_DIR = f"{TT_METAL_PATH}/third_party/lfs/profiler/tests/golden/device/logs"
TEST_DEVICE_LOGS = "test_device_logs.py"

TEST_FILE_IMPORTS = """
# THIS FILE IS AUTO-GENERATED
# Refer to docs to learn how to genereate this file using populate_golden.py

from tt_metal.tools.profiler.tests.device_log_run import run_test
"""

TEST_FILE_METHOD = """
@run_test
def test_{}():
    pass
"""


@click.command()
@click.option("-w", "--wipe", default=False, is_flag=True, help="Wipe the golden outputs folder")
def main(wipe):
    if wipe:
        os.system(f"rm -rf {GOLDEN_OUTPUTS_DIR}")
        os.mkdir(f"{GOLDEN_OUTPUTS_DIR}")

    testNames = []
    for logPath in os.listdir(GOLDEN_LOGS_DIR):
        correctFormat = re.search("^profile_log_device_.*.csv$", logPath)
        if correctFormat:
            testNames.append((logPath, logPath.split(".")[0].split("profile_log_device_")[-1].lower()))

    with open(f"tests/{TEST_DEVICE_LOGS}", "w") as testsMethods:

        print(TEST_FILE_IMPORTS, file=testsMethods)

        for logPath, testName in testNames:
            if not os.path.isdir(f"{GOLDEN_OUTPUTS_DIR}/test_{testName}"):
                print(f"Generating {testName} ... ", end="", flush=True)

                os.mkdir(f"{GOLDEN_OUTPUTS_DIR}/test_{testName}")
                os.system(f"cp {GOLDEN_LOGS_DIR}/{logPath} {GOLDEN_OUTPUTS_DIR}/test_{testName}/profile_log_device.csv")

                ret = os.system(
                    f"./process_device_log.py -d {GOLDEN_OUTPUTS_DIR}/test_{testName}/profile_log_device.csv --no-artifacts --no-print-stats --no-webapp"
                )
                assert ret == 0

                os.system(f"cp output/*.* {GOLDEN_OUTPUTS_DIR}/test_{testName}/")
                beautify_tt_js_blob(f"{GOLDEN_OUTPUTS_DIR}/test_{testName}/")

                print(f"Generated")

            print(TEST_FILE_METHOD.format(testName), file=testsMethods)


if __name__ == "__main__":
    main()
