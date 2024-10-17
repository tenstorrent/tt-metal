# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os, sys
import json
import re
import inspect
import pytest

import pandas as pd

from tt_metal.tools.profiler.common import (
    TT_METAL_HOME,
    PROFILER_HOST_DEVICE_SYNC_INFO,
    PROFILER_SCRIPTS_ROOT,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_LOGS_DIR,
    clear_profiler_runtime_artifacts,
)

from models.utility_functions import skip_for_grayskull

PROG_EXMP_DIR = "programming_examples/profiler"


def run_device_profiler_test(testName=None, setup=False, slowDispatch=False):
    name = inspect.stack()[1].function
    testCommand = f"build/{PROG_EXMP_DIR}/{name}"
    if testName:
        testCommand = testName
    print("Running: " + testCommand)
    clear_profiler_runtime_artifacts()
    slowDispatchEnv = ""
    if slowDispatch:
        slowDispatchEnv = "TT_METAL_SLOW_DISPATCH_MODE=1 "
    profilerRun = os.system(f"cd {TT_METAL_HOME} && {slowDispatchEnv}{testCommand}")
    assert profilerRun == 0

    setupStr = ""
    if setup:
        setupStr = f"-s {name}"

    postProcessRun = os.system(
        f"cd {PROFILER_SCRIPTS_ROOT} && " f"./process_device_log.py {setupStr} --no-artifacts --no-print-stats"
    )

    assert postProcessRun == 0, f"Log process script crashed with exit code {postProcessRun}"

    devicesData = {}
    with open(f"{PROFILER_ARTIFACTS_DIR}/output/device/device_analysis_data.json", "r") as devicesDataJson:
        devicesData = json.load(devicesDataJson)

    return devicesData


def get_function_name():
    frame = inspect.currentframe()
    return frame.f_code.co_name


@skip_for_grayskull()
def test_multi_op():
    OP_COUNT = 1000
    RUN_COUNT = 2
    REF_COUNT_DICT = {
        "grayskull": [108 * OP_COUNT * RUN_COUNT, 88 * OP_COUNT * RUN_COUNT],
        "wormhole_b0": [72 * OP_COUNT * RUN_COUNT, 64 * OP_COUNT * RUN_COUNT, 56 * OP_COUNT * RUN_COUNT],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setup=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    statName = f"BRISC KERNEL_START->KERNEL_END"

    assert statName in stats.keys(), "Wrong device analysis format"
    assert stats[statName]["stats"]["Count"] in REF_COUNT_DICT[ENV_VAR_ARCH_NAME], "Wrong Marker Repeat count"


def test_custom_cycle_count_slow_dispatch():
    REF_CYCLE_COUNT_PER_LOOP = 52
    LOOP_COUNT = 2000
    REF_CYCLE_COUNT = REF_CYCLE_COUNT_PER_LOOP * LOOP_COUNT
    REF_CYCLE_COUNT_HIGH_MULTIPLIER = 10
    REF_CYCLE_COUNT_LOW_MULTIPLIER = 5

    REF_CYCLE_COUNT_MAX = REF_CYCLE_COUNT * REF_CYCLE_COUNT_HIGH_MULTIPLIER
    REF_CYCLE_COUNT_MIN = REF_CYCLE_COUNT // REF_CYCLE_COUNT_LOW_MULTIPLIER

    devicesData = run_device_profiler_test(setup=True, slowDispatch=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    for risc in ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]:
        statName = f"{risc} KERNEL_START->KERNEL_END"

        assert statName in stats.keys(), "Wrong device analysis format"
        assert stats[statName]["stats"]["Average"] < REF_CYCLE_COUNT_MAX, "Wrong cycle count, too high"
        assert stats[statName]["stats"]["Average"] > REF_CYCLE_COUNT_MIN, "Wrong cycle count, too low"


def test_custom_cycle_count():
    REF_CYCLE_COUNT_PER_LOOP = 52
    LOOP_COUNT = 2000
    REF_CYCLE_COUNT = REF_CYCLE_COUNT_PER_LOOP * LOOP_COUNT
    REF_CYCLE_COUNT_HIGH_MULTIPLIER = 10
    REF_CYCLE_COUNT_LOW_MULTIPLIER = 5

    REF_CYCLE_COUNT_MAX = REF_CYCLE_COUNT * REF_CYCLE_COUNT_HIGH_MULTIPLIER
    REF_CYCLE_COUNT_MIN = REF_CYCLE_COUNT // REF_CYCLE_COUNT_LOW_MULTIPLIER

    devicesData = run_device_profiler_test(setup=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    for risc in ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]:
        statName = f"{risc} KERNEL_START->KERNEL_END"

        assert statName in stats.keys(), "Wrong device analysis format"
        assert stats[statName]["stats"]["Average"] < REF_CYCLE_COUNT_MAX, "Wrong cycle count, too high"
        assert stats[statName]["stats"]["Average"] > REF_CYCLE_COUNT_MIN, "Wrong cycle count, too low"


def test_full_buffer():
    OP_COUNT = 26
    RISC_COUNT = 5
    ZONE_COUNT = 125
    REF_COUNT_DICT = {
        "grayskull": [108 * OP_COUNT * RISC_COUNT * ZONE_COUNT, 88 * OP_COUNT * RISC_COUNT * ZONE_COUNT],
        "wormhole_b0": [
            72 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
            64 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
            56 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
        ],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setup=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]
    statName = "Marker Repeat"
    statNameEth = "Marker Repeat ETH"

    assert statName in stats.keys(), "Wrong device analysis format"

    if statNameEth in stats.keys():
        assert (
            stats[statName]["stats"]["Count"] - stats[statNameEth]["stats"]["Count"]
            in REF_COUNT_DICT[ENV_VAR_ARCH_NAME]
        ), "Wrong Marker Repeat count"
        assert stats[statNameEth]["stats"]["Count"] > 0, "Wrong Eth Marker Repeat count"
        assert stats[statNameEth]["stats"]["Count"] % (OP_COUNT * ZONE_COUNT) == 0, "Wrong Eth Marker Repeat count"
    else:
        assert stats[statName]["stats"]["Count"] in REF_COUNT_DICT[ENV_VAR_ARCH_NAME], "Wrong Marker Repeat count"


def test_dispatch_cores():
    OP_COUNT = 1
    RISC_COUNT = 1
    ZONE_COUNT = 37
    REF_COUNT_DICT = {
        "grayskull": {
            "Tensix CQ Dispatch": 16,
            "Tensix CQ Prefetch": 24,
        },
        "wormhole_b0": {
            "Tensix CQ Dispatch": 16,
            "Tensix CQ Prefetch": 24,
        },
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "1"

    devicesData = run_device_profiler_test(setup=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    verifiedStat = []
    for stat in REF_COUNT_DICT[ENV_VAR_ARCH_NAME].keys():
        if stat in stats.keys():
            verifiedStat.append(stat)
            assert stats[stat]["stats"]["Count"] == REF_COUNT_DICT[ENV_VAR_ARCH_NAME][stat], "Wrong Dispatch zone count"

    statTypes = ["Dispatch", "Prefetch"]
    statTypesSet = set(statTypes)
    for statType in statTypes:
        for stat in verifiedStat:
            if statType in stat:
                statTypesSet.remove(statType)
    assert len(statTypesSet) == 0
    os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"


def test_profiler_host_device_sync():
    TOLERANCE = 0.1

    os.environ["TT_METAL_PROFILER_SYNC"] = "1"
    syncInfoFile = PROFILER_LOGS_DIR / PROFILER_HOST_DEVICE_SYNC_INFO

    deviceData = run_device_profiler_test(testName="pytest ./tests/ttnn/tracy/test_profiler_sync.py::test_all_devices")
    reportedFreq = deviceData["data"]["deviceInfo"]["freq"] * 1e6
    assert os.path.isfile(syncInfoFile)

    syncinfoDF = pd.read_csv(syncInfoFile)
    devices = sorted(syncinfoDF["device id"].unique())
    for device in devices:
        freq = float(syncinfoDF[syncinfoDF["device id"] == device].iloc[-1]["frequency"]) * 1e9

        assert freq < (reportedFreq * (1 + TOLERANCE)), f"Frequency too large on device {device}"
        assert freq > (reportedFreq * (1 - TOLERANCE)), f"Frequency too small on device {device}"

    deviceData = run_device_profiler_test(testName="pytest ./tests/ttnn/tracy/test_profiler_sync.py::test_with_ops")
    reportedFreq = deviceData["data"]["deviceInfo"]["freq"] * 1e6
    assert os.path.isfile(syncInfoFile)

    syncinfoDF = pd.read_csv(syncInfoFile)
    devices = sorted(syncinfoDF["device id"].unique())
    for device in devices:
        freq = float(syncinfoDF[syncinfoDF["device id"] == device].iloc[-1]["frequency"]) * 1e9

        assert freq < (reportedFreq * (1 + TOLERANCE)), f"Frequency too large on device {device}"
        assert freq > (reportedFreq * (1 - TOLERANCE)), f"Frequency too small on device {device}"
