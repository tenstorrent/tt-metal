import os, sys
import json
import re
import inspect
import pytest


import tests.tt_metal.tools.profiler.common as common

REPO_PATH = common.get_repo_path()
TT_METAL_PATH = f"{REPO_PATH}/tt_metal"
PROFILER_DIR = f"{TT_METAL_PATH}/tools/profiler/"
PROFILER_OUT_DIR = f"{PROFILER_DIR}/output/device"
GS_PROG_EXMP_DIR = "programming_examples/profiler/device/grayskull"


def run_device_profiler_test(doubleRun=False, setup=False):
    name = inspect.stack()[1].function
    profilerRun = os.system(
        f"cd {REPO_PATH} && " f"make {GS_PROG_EXMP_DIR}/{name} && " f"build/{GS_PROG_EXMP_DIR}/{name}"
    )
    assert profilerRun == 0

    if doubleRun:
        # Run test under twice to make sure icache is populated
        # with instructions for test
        profilerRun = os.system(f"cd {REPO_PATH} && " f"build/{GS_PROG_EXMP_DIR}/{name}")
        assert profilerRun == 0

    setupStr = ""
    if setup:
        setupStr = f"-s {name}"

    postProcessRun = os.system(
        f"cd {PROFILER_DIR} && " f"./process_device_log.py {setupStr} --no-artifacts --no-print-stats --no-webapp"
    )

    assert postProcessRun == 0, f"Log process script crashed with exit code {postProcessRun}"

    devicesData = {}
    with open(f"{PROFILER_OUT_DIR}/device_analysis_data.json", "r") as devicesDataJson:
        devicesData = json.load(devicesDataJson)

    return devicesData


def get_function_name():
    frame = inspect.currentframe()
    return frame.f_code.co_name


# @pytest.mark.skip(reason="Some machines are acting flaky on this test")
def test_custom_cycle_count():
    REF_CYCLE_COUNT = 52
    REF_CYCLE_COUNT_MARGIN = 6
    REF_RISC_COUNT = 5

    REF_CYCLE_COUNT_MAX = REF_CYCLE_COUNT + REF_CYCLE_COUNT_MARGIN
    REF_CYCLE_COUNT_MIN = REF_CYCLE_COUNT - REF_CYCLE_COUNT_MARGIN

    devicesData = run_device_profiler_test(doubleRun=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]
    riscCount = 0
    for key in stats.keys():
        match = re.search(r"^.* kernel start -> .* kernel end$", key)
        if match:
            riscCount += 1
            assert stats[key]["stats"]["Range"] < REF_CYCLE_COUNT_MAX  , "Wrong cycle count, too high"
            assert stats[key]["stats"]["Range"] > REF_CYCLE_COUNT_MIN  , "Wrong cycle count, too low"
    assert riscCount == REF_RISC_COUNT, "Wrong RISC count"


def test_full_buffer():
    REF_COUNT = 4200  # 120(cores) x 5(riscs) x 7(buffer size in marker)
    REF_RISC_COUNT = 5

    devicesData = run_device_profiler_test(setup=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]
    riscCount = 0
    assert "Marker Repeat" in stats.keys(), "Wrong device analysis format"
    assert stats["Marker Repeat"]["stats"]["Count"] == REF_COUNT, "Wrong Marker Repeat count"
