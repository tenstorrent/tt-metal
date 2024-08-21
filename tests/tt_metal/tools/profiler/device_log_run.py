# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os, sys
from filecmp import dircmp, cmp
from pathlib import Path
from difflib import Differ
import re
import fileinput

import jsbeautifier
from loguru import logger

from tt_metal.tools.profiler.common import TT_METAL_HOME, PROFILER_SCRIPTS_ROOT, PROFILER_ARTIFACTS_DIR

TT_METAL_PATH = TT_METAL_HOME / "tt_metal"
GOLDEN_OUTPUTS_DIR = TT_METAL_PATH / "third_party/lfs/profiler/tests/golden/device/outputs"

RE_RANDOM_ID_STRINGS = [r'if \(document.getElementById\("{0}"\)\) {{', r'    Plotly.newPlot\("{0}", \[{{']
DIFF_LINE_COUNT_LIMIT = 40


def replace_random_id(line):
    for randomIDStr in RE_RANDOM_ID_STRINGS:
        match = re.search(f"^{randomIDStr.format('.*')}$", line)
        if match:
            return randomIDStr.format("random_id_replaced_for_automation").replace("\\", "")
    return line


def filter_device_analysis_data(testOutputFolder):
    testFiles = os.scandir(testOutputFolder)
    for testFile in testFiles:
        if "device_analysis_data.json" in testFile.name:
            testFilePath = f"{testOutputFolder}/{testFile.name}"
            for line in fileinput.input(testFilePath, inplace=True):
                if "deviceInputLog" not in line:
                    print(line, end="")


def run_device_log_compare_golden(test):
    goldenPath = GOLDEN_OUTPUTS_DIR / test
    underTestPath = PROFILER_ARTIFACTS_DIR / "output/device"

    ret = os.system(
        f"cd {PROFILER_SCRIPTS_ROOT} && ./process_device_log.py -d {goldenPath}/profile_log_device.csv --no-print-stats --no-artifacts"
    )
    assert ret == 0, f"Log process script crashed with exit code {ret}"

    filter_device_analysis_data(underTestPath)

    dcmp = dircmp(goldenPath, underTestPath)

    for diffFile in dcmp.diff_files:
        goldenFile = Path(f"{goldenPath}/{diffFile}")
        underTestFile = Path(f"{underTestPath}/{diffFile}")

        diffStr = f"\n{diffFile}\n"
        with open(goldenFile) as golden, open(underTestFile) as underTest:
            differ = Differ()
            lineCount = 0
            for line in differ.compare(golden.readlines(), underTest.readlines()):
                if line[0] in ["-", "+", "?"]:
                    diffStr += line
                    lineCount += 1
                if lineCount > DIFF_LINE_COUNT_LIMIT:
                    diffStr += (
                        "[NOTE: limited lines on log output, run locally without line count limits for more info]"
                    )
                    break
        logger.error(diffStr)

    assert not dcmp.diff_files, f"{dcmp.diff_files} cannot be different from golden"
    assert not dcmp.right_only, f"New output files: {dcmp.right_only}"
    assert not dcmp.left_only, f"Golden files not present in output: {dcmp.left_only}"
    assert not dcmp.funny_files, f"Unreadable files: {dcmp.funny_files}"


def run_test(func):
    def test():
        run_device_log_compare_golden(func.__name__)

    return test
