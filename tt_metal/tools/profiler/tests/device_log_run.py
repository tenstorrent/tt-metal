import os, sys
from filecmp import dircmp, cmp
from pathlib import Path
from difflib import Differ
import re

import jsbeautifier

try:
    REPO_PATH = os.environ["PYTHONPATH"]
except KeyError:
    print("PYTHONPATH has to be setup. Please refer to gettins started docs", file=sys.stderr)


TT_METAL_PATH = f"{REPO_PATH}/tt_metal"
GOLDEN_OUTPUTS_DIR = f"{TT_METAL_PATH}/third_party/lfs/profiler/tests/golden/device/outputs"

RE_RANDOM_ID_STRINGS = [
'if \(document.getElementById\("{0}"\)\) {{',
'    Plotly.newPlot\("{0}", \[{{'
]

def replace_random_id(line):
    for randomIDStr in RE_RANDOM_ID_STRINGS:
        match = re.search(f"^{randomIDStr.format('.*')}$", line)
        if match:
            return randomIDStr.format('random_id_replaced_for_automation').replace('\\','')
    return line



def beautify_tt_js_blob(testOutputFoler):
    testFiles = os.scandir(testOutputFoler)
    for testFile in testFiles:
        if ".html" in testFile.name:
            testFilePath = f"{testOutputFoler}/{testFile.name}"
            with open(testFilePath) as htmlFile:
                for htmlLine in htmlFile.readlines():
                    if "!function" in htmlLine:
                        jsBlobs = htmlLine.rsplit("script", 2)
                        assert len(jsBlobs) > 2
                        jsBlob = jsBlobs[1].strip(' <>/"')
                        beautyJS = jsbeautifier.beautify(jsBlob)
                        break

            os.remove(testFilePath)

            beautyJS = beautyJS.split("\n")
            jsLines = []
            for jsLine in beautyJS:
                jsLines.append (replace_random_id(jsLine))

            beautyJS = "\n".join(jsLines)

            with open(f"{testFilePath.split('.html')[0]}.js", "w") as jsFile:
                jsFile.writelines(beautyJS)




def run_device_log_compare_golden(test):
    goldenPath = f"{GOLDEN_OUTPUTS_DIR}/{test}"
    underTestPath = "output"

    ret = os.system(
        f"./process_device_log.py -d {goldenPath}/profile_log_device.csv --no-print-stats --no-artifacts --no-webapp"
    )
    assert ret == 0

    beautify_tt_js_blob(underTestPath)

    dcmp = dircmp(goldenPath, underTestPath)

    # Files that are not expected to be different except for allowed phrases
    for diffFile in dcmp.diff_files:
        goldenFile = Path(f"{goldenPath}/{diffFile}")
        underTestFile = Path(f"{underTestPath}/{diffFile}")

        with open(goldenFile) as golden, open(underTestFile) as underTest:
            differ = Differ()
            print()
            for count, line in enumerate(differ.compare(golden.readlines(), underTest.readlines())):
                if line[0] in ["-", "+", "?"]:
                    print(count, line, end="")
        assert not trueDiff, f"{diffFile} cannot be different from golden"

    assert not (dcmp.right_only or dcmp.left_only or dcmp.funny_files), (
        dcmp.right_only,
        dcmp.left_only,
        dcmp.funny_files,
    )


def run_test(func):
    def test():
        run_device_log_compare_golden(func.__name__)

    return test
