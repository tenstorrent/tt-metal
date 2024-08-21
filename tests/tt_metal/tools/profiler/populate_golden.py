#! /usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import os, sys
import csv
import re

import click
from loguru import logger

from device_log_run import filter_device_analysis_data

from tt_metal.tools.profiler.common import TT_METAL_HOME, PROFILER_SCRIPTS_ROOT, PROFILER_ARTIFACTS_DIR, rm

TT_METAL_PATH = TT_METAL_HOME / "tt_metal"
GOLDEN_OUTPUTS_DIR = TT_METAL_PATH / "third_party/lfs/profiler/tests/golden/device/outputs"
GOLDEN_LOGS_DIR = TT_METAL_PATH / "third_party/lfs/profiler/tests/golden/device/logs"
TEST_DEVICE_LOGS_PATH = TT_METAL_HOME / "tests/tt_metal/tools/profiler/test_device_logs.py"

TEST_FILE_IMPORTS = """# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# THIS FILE IS AUTO-GENERATED
# Refer to the profiler README to learn how to generate this file using populate_golden.py

from tests.tt_metal.tools.profiler.device_log_run import run_test
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
        rm(GOLDEN_OUTPUTS_DIR)
        os.mkdir(f"{GOLDEN_OUTPUTS_DIR}")

    testNames = []
    for logPath in os.listdir(GOLDEN_LOGS_DIR):
        correctFormat = re.search("^profile_log_device_.*.csv$", logPath)
        if correctFormat:
            testName = logPath.split(".")[0].split("profile_log_device_")[-1].lower()
            testNames.append((logPath, testName))
    testNames.sort()

    with open(f"{TEST_DEVICE_LOGS_PATH}", "w") as testsMethods:
        print(TEST_FILE_IMPORTS, file=testsMethods)

        for testNum, (logPath, testName) in enumerate(testNames):
            if not os.path.isdir(f"{GOLDEN_OUTPUTS_DIR}/test_{testName}"):
                logger.info(f"Generating {testName}")

                csvPath = f"{GOLDEN_LOGS_DIR}/{logPath}"

                csvIsOld = False
                lines = []
                with open(csvPath, "r") as deviceLog:
                    lines = deviceLog.readlines()
                    if "stat value" not in lines[1]:
                        csvIsOld = True

                if csvIsOld:
                    infoHead = lines[0]
                    csvHead = "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], stat value, Run ID, zone name, zone phase, source line, source file\n"
                    lines = lines[1:]
                    with open(csvPath, "w") as deviceLog:
                        for line in lines:
                            deviceLog.write(line)

                    with open(csvPath, "r") as deviceLog:
                        csvRows = csv.DictReader(deviceLog)
                        with open("./tmp.csv", "w") as tmpLog:
                            tmpLog.write(infoHead)
                            tmpLog.write(csvHead)
                            newFields = csvRows.fieldnames
                            newFields = newFields[:5] + [" stat value"] + newFields[5:]
                            csvTmp = csv.DictWriter(tmpLog, newFields)

                            for row in csvRows:
                                row[" stat value"] = 0
                                csvTmp.writerow(row)

                    os.system(f"cp tmp.csv {csvPath}")

                os.mkdir(f"{GOLDEN_OUTPUTS_DIR}/test_{testName}")
                os.system(f"cp {GOLDEN_LOGS_DIR}/{logPath} {GOLDEN_OUTPUTS_DIR}/test_{testName}/profile_log_device.csv")

                ret = os.system(
                    f"cd {PROFILER_SCRIPTS_ROOT} && ./process_device_log.py -d {GOLDEN_OUTPUTS_DIR}/test_{testName}/profile_log_device.csv --no-artifacts --no-print-stats"
                )
                assert ret == 0, f"Log process script crashed with exit code {ret}"

                os.system(f"cp {PROFILER_ARTIFACTS_DIR}/output/device/*.* {GOLDEN_OUTPUTS_DIR}/test_{testName}/")
                filter_device_analysis_data(f"{GOLDEN_OUTPUTS_DIR}/test_{testName}/")

            # Remove line ending from the last test
            if testNum == (len(testNames) - 1):
                print(TEST_FILE_METHOD[:-1].format(testName), file=testsMethods)
            else:
                print(TEST_FILE_METHOD.format(testName), file=testsMethods)


if __name__ == "__main__":
    main()
