# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os, sys
import json
import re
import inspect
import pytest
import subprocess
import glob
from loguru import logger

import pandas as pd
import numpy as np

from tt_metal.tools.profiler.common import (
    TT_METAL_HOME,
    PROFILER_HOST_DEVICE_SYNC_INFO,
    PROFILER_SCRIPTS_ROOT,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_LOGS_DIR,
    clear_profiler_runtime_artifacts,
)

from models.utility_functions import skip_for_grayskull, skip_for_blackhole

PROG_EXMP_DIR = "programming_examples/profiler"
TRACY_TESTS_DIR = "./tests/ttnn/tracy"


def get_device_data(setupStr=""):
    postProcessRun = os.system(
        f"cd {PROFILER_SCRIPTS_ROOT} && " f"./process_device_log.py {setupStr} --no-artifacts --no-print-stats"
    )

    assert postProcessRun == 0, f"Log process script crashed with exit code {postProcessRun}"

    devicesData = {}
    with open(f"{PROFILER_ARTIFACTS_DIR}/output/device/device_analysis_data.json", "r") as devicesDataJson:
        devicesData = json.load(devicesDataJson)

    return devicesData


def set_env_vars(**kwargs):
    envVarsDict = {
        "doSync": "TT_METAL_PROFILER_SYNC=1 ",
        "doDispatchCores": "TT_METAL_DEVICE_PROFILER_DISPATCH=1 ",
        "slowDispatch": "TT_METAL_SLOW_DISPATCH_MODE=1 ",
        "dispatchFromEth": "WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ",
        "enable_noc_tracing": "TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 ",
    }
    envVarsStr = " "
    for arg, argVal in kwargs.items():
        if argVal:
            envVarsStr += envVarsDict[arg]
    return envVarsStr


# returns True if test passed, False if test was SKIPPED
def run_gtest_profiler_test(testbin, testname, doSync=False, enable_noc_tracing=False, skip_get_device_data=False):
    clear_profiler_runtime_artifacts()
    envVars = set_env_vars(doSync=doSync, enable_noc_tracing=enable_noc_tracing)
    testCommand = f"cd {TT_METAL_HOME} && {envVars} {testbin} --gtest_filter={testname}"
    print()
    logger.info(f"Running: {testCommand}")
    output = subprocess.check_output(testCommand, stderr=subprocess.STDOUT, shell=True).decode("UTF-8")
    print(output)
    if "SKIPPED" not in output:
        if not skip_get_device_data:
            get_device_data()
        return True
    else:
        return False


def run_device_profiler_test(
    testName=None,
    setupAutoExtract=False,
    slowDispatch=False,
    doSync=False,
    doDispatchCores=False,
    dispatchFromEth=False,
):
    name = inspect.stack()[1].function
    testCommand = f"build/{PROG_EXMP_DIR}/{name}"
    if testName:
        testCommand = testName
    clear_profiler_runtime_artifacts()
    envVars = set_env_vars(
        slowDispatch=slowDispatch, doSync=doSync, doDispatchCores=doDispatchCores, dispatchFromEth=dispatchFromEth
    )
    testCommand = f"cd {TT_METAL_HOME} && {envVars} {testCommand}"
    print()
    logger.info(f"Running: {testCommand}")
    profilerRun = os.system(testCommand)
    assert profilerRun == 0

    setupStr = ""
    if setupAutoExtract:
        setupStr = f"-s {name}"

    return get_device_data(setupStr)


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
        "blackhole": [130 * OP_COUNT * RUN_COUNT, 120 * OP_COUNT * RUN_COUNT, 110 * OP_COUNT * RUN_COUNT],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setupAutoExtract=True)

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

    devicesData = run_device_profiler_test(setupAutoExtract=True, slowDispatch=True)

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

    devicesData = run_device_profiler_test(setupAutoExtract=True)

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
        "blackhole": [
            130 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
            120 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
            110 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
        ],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setupAutoExtract=True)

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


@skip_for_blackhole()
def test_dispatch_cores():
    OP_COUNT = 1
    RISC_COUNT = 1
    ZONE_COUNT = 37
    REF_COUNT_DICT = {
        "Tensix CQ Dispatch": [600, 760, 1310, 2330],
        "Tensix CQ Prefetch": [900, 1440, 3870, 5000],
        "dispatch_total_cq_cmd_op_time": [103],
        "dispatch_go_send_wait_time": [103],
    }

    def verify_stats(devicesData, statTypes, allowedRange):
        verifiedStat = []
        for device, deviceData in devicesData["data"]["devices"].items():
            for ref, counts in REF_COUNT_DICT.items():
                if ref in deviceData["cores"]["DEVICE"]["analysis"].keys():
                    verifiedStat.append(ref)
                    res = False
                    readCount = deviceData["cores"]["DEVICE"]["analysis"][ref]["stats"]["Count"]
                    for count in counts:
                        if count - allowedRange <= readCount <= count + allowedRange:
                            res = True
                            break
                    assert (
                        res
                    ), f"Wrong tensix dispatch zone count for {ref}, read {readCount} which is not within {allowedRange} cycle counts of any of the limits {counts}"

        statTypesSet = set(statTypes)
        for statType in statTypes:
            for stat in verifiedStat:
                if statType in stat and statType in statTypesSet:
                    statTypesSet.remove(statType)
        assert len(statTypesSet) == 0

    verify_stats(
        run_device_profiler_test(setupAutoExtract=True, doDispatchCores=True),
        statTypes=["Dispatch", "Prefetch"],
        allowedRange=150,
    )

    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_with_ops",
            setupAutoExtract=True,
            doDispatchCores=True,
        ),
        statTypes=["Dispatch", "Prefetch"],
        allowedRange=1000,
    )

    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_all_devices",
            setupAutoExtract=True,
            doDispatchCores=True,
        ),
        statTypes=["Dispatch", "Prefetch"],
        allowedRange=1000,
    )

    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_trace_runs.py",
            setupAutoExtract=False,
            doDispatchCores=True,
        ),
        statTypes=["dispatch_total_cq_cmd_op_time", "dispatch_go_send_wait_time"],
        allowedRange=0,  # This test is basically counting ops and should be exact regardless of changes to dispatch code or harvesting.
    )


# Eth dispatch will be deprecated
@skip_for_blackhole()
@skip_for_grayskull()
def test_ethernet_dispatch_cores():
    REF_COUNT_DICT = {"Ethernet CQ Dispatch": [590, 1220, 1430, 1660, 2320], "Ethernet CQ Prefetch": [1180, 4630]}
    devicesData = run_device_profiler_test(
        testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_with_ops",
        setupAutoExtract=True,
        doDispatchCores=True,
        dispatchFromEth=True,
    )
    for device, deviceData in devicesData["data"]["devices"].items():
        for ref, counts in REF_COUNT_DICT.items():
            if ref in deviceData["cores"]["DEVICE"]["analysis"].keys():
                res = False
                readCount = deviceData["cores"]["DEVICE"]["analysis"][ref]["stats"]["Count"]
                allowedRange = 200
                for count in counts:
                    if count - allowedRange < readCount < count + allowedRange:
                        res = True
                        break
                assert (
                    res
                ), f"Wrong ethernet dispatch zone count for {ref}, read {readCount} which is not within {allowedRange} cycle counts of any of the limits {counts}"

    devicesData = run_device_profiler_test(
        testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_all_devices",
        setupAutoExtract=True,
        doDispatchCores=True,
        dispatchFromEth=True,
    )
    for device, deviceData in devicesData["data"]["devices"].items():
        for ref, counts in REF_COUNT_DICT.items():
            if ref in deviceData["cores"]["DEVICE"]["analysis"].keys():
                res = False
                readCount = deviceData["cores"]["DEVICE"]["analysis"][ref]["stats"]["Count"]
                allowedRange = 200
                for count in counts:
                    if count - allowedRange < readCount < count + allowedRange:
                        res = True
                        break
                assert (
                    res
                ), f"Wrong ethernet dispatch zone count for {ref}, read {readCount} which is not within {allowedRange} cycle counts of any of the limits {counts}"


@skip_for_grayskull()
def test_profiler_host_device_sync():
    TOLERANCE = 0.1

    syncInfoFile = PROFILER_LOGS_DIR / PROFILER_HOST_DEVICE_SYNC_INFO

    deviceData = run_device_profiler_test(
        testName=f"pytest {TRACY_TESTS_DIR}/test_profiler_sync.py::test_all_devices", doSync=True
    )
    reportedFreq = deviceData["data"]["deviceInfo"]["freq"] * 1e6
    assert os.path.isfile(syncInfoFile)

    syncinfoDF = pd.read_csv(syncInfoFile)
    devices = sorted(syncinfoDF["device id"].unique())
    for device in devices:
        deviceFreq = syncinfoDF[syncinfoDF["device id"] == device].iloc[-1]["frequency"]
        if not np.isnan(deviceFreq):  # host sync entry
            freq = float(deviceFreq) * 1e9

            assert freq < (reportedFreq * (1 + TOLERANCE)), f"Frequency {freq} is too large on device {device}"
            assert freq > (reportedFreq * (1 - TOLERANCE)), f"Frequency {freq} is too small on device {device}"
        else:  # device sync entry
            deviceFreqRatio = syncinfoDF[syncinfoDF["device id"] == device].iloc[-1]["device_frequency_ratio"]
            assert deviceFreqRatio < (
                1 + TOLERANCE
            ), f"Frequency ratio {deviceFreqRatio} is too large on device {device}"
            assert deviceFreqRatio > (
                1 - TOLERANCE
            ), f"Frequency ratio {deviceFreqRatio} is too small on device {device}"

    deviceData = run_device_profiler_test(
        testName=f"pytest {TRACY_TESTS_DIR}/test_profiler_sync.py::test_with_ops", doSync=1
    )
    reportedFreq = deviceData["data"]["deviceInfo"]["freq"] * 1e6
    assert os.path.isfile(syncInfoFile)

    syncinfoDF = pd.read_csv(syncInfoFile)
    devices = sorted(syncinfoDF["device id"].unique())
    for device in devices:
        deviceFreq = syncinfoDF[syncinfoDF["device id"] == device].iloc[-1]["frequency"]
        if not np.isnan(deviceFreq):  # host sync entry
            freq = float(deviceFreq) * 1e9

            assert freq < (reportedFreq * (1 + TOLERANCE)), f"Frequency {freq} is too large on device {device}"
            assert freq > (reportedFreq * (1 - TOLERANCE)), f"Frequency {freq} is too small on device {device}"


def test_timestamped_events():
    OP_COUNT = 2
    RISC_COUNT = 5
    ZONE_COUNT = 100
    WH_ERISC_COUNTS = [0, 3, 6]  # N150, N300, T3K
    WH_TENSIX_COUNTS = [72, 64, 56]
    BH_ERISC_COUNTS = [0, 1, 6, 8]
    BH_TENSIX_COUNTS = [130, 120, 110]

    WH_COMBO_COUNTS = []
    for T in WH_TENSIX_COUNTS:
        for E in WH_ERISC_COUNTS:
            WH_COMBO_COUNTS.append((T, E))

    BH_COMBO_COUNTS = []
    for T in BH_TENSIX_COUNTS:
        for E in BH_ERISC_COUNTS:
            BH_COMBO_COUNTS.append((T, E))

    REF_COUNT_DICT = {
        "grayskull": [108 * OP_COUNT * RISC_COUNT * ZONE_COUNT, 88 * OP_COUNT * RISC_COUNT * ZONE_COUNT],
        "wormhole_b0": [(T * RISC_COUNT + E) * OP_COUNT * ZONE_COUNT for T, E in WH_COMBO_COUNTS],
        "blackhole": [(T * RISC_COUNT + E) * OP_COUNT * ZONE_COUNT for T, E in BH_COMBO_COUNTS],
    }
    REF_ERISC_COUNT = {
        "wormhole_b0": [C * OP_COUNT * ZONE_COUNT for C in WH_ERISC_COUNTS],
        "blackhole": [C * OP_COUNT * ZONE_COUNT for C in BH_ERISC_COUNTS],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setupAutoExtract=True)

    if ENV_VAR_ARCH_NAME in REF_ERISC_COUNT.keys():
        eventCount = len(
            devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["riscs"]["TENSIX"]["events"]["erisc_events"]
        )
        assert eventCount in REF_ERISC_COUNT[ENV_VAR_ARCH_NAME], "Wrong erisc event count"

    if ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys():
        eventCount = len(
            devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["riscs"]["TENSIX"]["events"]["all_events"]
        )
        assert eventCount in REF_COUNT_DICT[ENV_VAR_ARCH_NAME], "Wrong event count"


def test_noc_event_profiler_linked_multicast_hang():
    # test that we can avoid hangs with linked multicast
    # see tt-metal issue #22578
    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in ["grayskull", "wormhole_b0", "blackhole"]

    testCommand = "build/test/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency"
    # note: this runs a long series repeated multicasts from worker {1,1} to grid {2,2},{3,3}
    # note: -m6 is multicast test mode, -link activates linked multicast
    testCommandArgs = "-tx 3 -ty 3 -sx 2 -sy 2 -rx 1 -ry 1 -m 6 -link -profdump"
    clear_profiler_runtime_artifacts()
    nocEventProfilerEnv = "TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1"
    profilerRun = os.system(f"cd {TT_METAL_HOME} && {nocEventProfilerEnv} {testCommand} {testCommandArgs}")
    assert profilerRun == 0

    expected_trace_file = f"{PROFILER_LOGS_DIR}/noc_trace_dev0_ID0.json"
    assert os.path.isfile(expected_trace_file)

    with open(expected_trace_file, "r") as nocTraceJson:
        noc_trace_data = json.load(nocTraceJson)


def test_noc_event_profiler():
    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in ["grayskull", "wormhole_b0", "blackhole"]

    testCommand = f"build/{PROG_EXMP_DIR}/test_noc_event_profiler"
    clear_profiler_runtime_artifacts()
    nocEventProfilerEnv = "TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1"
    profilerRun = os.system(f"cd {TT_METAL_HOME} && {nocEventProfilerEnv} {testCommand}")
    assert profilerRun == 0

    expected_trace_file = f"{PROFILER_LOGS_DIR}/noc_trace_dev0_ID0.json"
    assert os.path.isfile(expected_trace_file)

    with open(expected_trace_file, "r") as nocTraceJson:
        noc_trace_data = json.load(nocTraceJson)
        assert len(noc_trace_data) == 8


@skip_for_blackhole()
def test_fabric_event_profiler_unicast():
    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in ["wormhole_b0", "blackhole"]

    # test that current device has a valid fabric API connection
    sanity_check_test_bin = "build/test/tt_metal/tt_fabric/fabric_unit_tests"
    sanity_check_test_name = "Fabric1DFixture.TestUnicastConnAPI"
    sanity_check_succeeded = run_gtest_profiler_test(
        sanity_check_test_bin, sanity_check_test_name, skip_get_device_data=True
    )
    if not sanity_check_succeeded:
        logger.info("Device does not have testable fabric connections, skipping ...")
        return

    # if device supports fabric API, test fabric event profiler
    test_bin = "build/test/tt_metal/tt_fabric/fabric_unit_tests"
    test_name = "Fabric1DFixture.TestUnicastRawWithTracing"
    nocEventProfilerEnv = "TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1"
    try:
        not_skipped = run_gtest_profiler_test(test_bin, test_name, False, True)
        assert not_skipped, f"gtest command '{test_bin}' was skipped unexpectedly"
    except subprocess.CalledProcessError as e:
        ret_code = e.returncode
        assert ret_code == 0, f"test command '{test_bin}' returned unsuccessfully"

    expected_cluster_coords_file = f"{PROFILER_LOGS_DIR}/cluster_coordinates.json"
    assert os.path.isfile(
        expected_cluster_coords_file
    ), f"expected cluster coordinates file '{expected_cluster_coords_file}' does not exist"

    expected_trace_file = f"{PROFILER_LOGS_DIR}/noc_trace_dev0_ID0.json"
    assert os.path.isfile(expected_trace_file), f"expected noc trace file '{expected_trace_file}' does not exist"

    fabric_event_count = 0
    with open(expected_trace_file, "r") as nocTraceJson:
        try:
            noc_trace_data = json.load(nocTraceJson)
        except json.JSONDecodeError:
            raise ValueError(f"noc trace file '{expected_trace_file}' is not a valid JSON file")

        assert isinstance(noc_trace_data, list), f"noc trace file '{expected_trace_file}' format is incorrect"
        assert len(noc_trace_data) > 0, f"noc trace file '{expected_trace_file}' is empty"
        for event in noc_trace_data:
            assert isinstance(event, dict), f"noc trace file format error; found event that is not a dict"
            if event.get("type", "") == "FABRIC_UNICAST_WRITE":
                fabric_event_count += 1
                fabric_send_metadata = event.get("fabric_send", None)
                if fabric_send_metadata is not None:
                    assert fabric_send_metadata.get("eth_chan", None) is not None
                    assert fabric_send_metadata.get("start_distance", None) is not None

    EXPECTED_FABRIC_EVENT_COUNT = 10
    assert (
        fabric_event_count == EXPECTED_FABRIC_EVENT_COUNT
    ), f"Incorrect number of fabric events found in noc trace: {fabric_event_count}, expected {EXPECTED_FABRIC_EVENT_COUNT}"


@skip_for_blackhole()
def test_fabric_event_profiler_1d_multicast():
    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in ["wormhole_b0", "blackhole"]

    # test that current device has a valid fabric API connection
    sanity_check_test_bin = "build/test/tt_metal/tt_fabric/fabric_unit_tests"
    sanity_check_test_name = "Fabric1DFixture.TestMCastConnAPI"
    sanity_check_succeeded = run_gtest_profiler_test(
        sanity_check_test_bin, sanity_check_test_name, skip_get_device_data=True
    )
    if not sanity_check_succeeded:
        logger.info("Device does not have testable fabric connections, skipping ...")
        return

    # if device supports fabric API, test fabric event profiler
    test_bin = "build/test/tt_metal/tt_fabric/fabric_unit_tests"
    tests = ["Fabric1DFixture.TestChipMCast1DWithTracing", "Fabric1DFixture.TestChipMCast1DWithTracing2"]
    expected_outputs = [
        {"START_DISTANCE": 1, "RANGE": 3, "FABRIC_EVENT_COUNT": 100},
        {"START_DISTANCE": 2, "RANGE": 2, "FABRIC_EVENT_COUNT": 100},
    ]

    for test_name, expected_output in zip(tests, expected_outputs):
        nocEventProfilerEnv = "TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1"
        try:
            not_skipped = run_gtest_profiler_test(test_bin, test_name, False, True)
            assert not_skipped, f"gtest command '{test_bin}' was skipped unexpectedly"
        except subprocess.CalledProcessError as e:
            ret_code = e.returncode
            assert ret_code == 0, f"test command '{test_bin}' returned unsuccessfully"

        expected_cluster_coords_file = f"{PROFILER_LOGS_DIR}/cluster_coordinates.json"
        assert os.path.isfile(
            expected_cluster_coords_file
        ), f"expected cluster coordinates file '{expected_cluster_coords_file}' does not exist"

        noc_trace_files = glob.glob(f"{PROFILER_LOGS_DIR}/noc_trace_dev[0-9]_ID[0-9].json")

        fabric_event_count = 0
        for trace_file in noc_trace_files:
            with open(trace_file, "r") as nocTraceJson:
                try:
                    noc_trace_data = json.load(nocTraceJson)
                except json.JSONDecodeError:
                    raise ValueError(f"noc trace file '{trace_file}' is not a valid JSON file")

                assert isinstance(noc_trace_data, list), f"noc trace file '{trace_file}' format is incorrect"
                assert len(noc_trace_data) > 0, f"noc trace file '{trace_file}' is empty"
                for event in noc_trace_data:
                    assert isinstance(event, dict), f"noc trace file format error; found event that is not a dict"
                    if event.get("type", "") == "FABRIC_UNICAST_WRITE":
                        fabric_event_count += 1
                        fabric_send_metadata = event.get("fabric_send", None)
                        if fabric_send_metadata is not None:
                            assert fabric_send_metadata.get("eth_chan", None) is not None
                            assert (
                                fabric_send_metadata.get("start_distance", None) == expected_output["START_DISTANCE"]
                            ), f"Incorrect start distance for fabric event in noc trace: {fabric_send_metadata.get('start_distance', None)}, "
                            f"expected {expected_output['START_DISTANCE']}"
                            assert (
                                fabric_send_metadata.get("range", None) == expected_output["RANGE"]
                            ), f"Incorrect range for fabric event in noc trace: {fabric_send_metadata.get('range', None)}, "
                            f"expected {expected_output['RANGE']}"

        assert (
            fabric_event_count == expected_output["FABRIC_EVENT_COUNT"]
        ), f"Incorrect number of fabric events found in noc trace: {fabric_event_count}, expected {EXPECTED_FABRIC_EVENT_COUNT}"


def test_sub_device_profiler():
    ARCH_NAME = os.getenv("ARCH_NAME")
    run_gtest_profiler_test(
        "./build/test/tt_metal/unit_tests_dispatch",
        "CommandQueueSingleCardFixture.TensixTestSubDeviceBasicPrograms",
    )
    run_gtest_profiler_test(
        "./build/test/tt_metal/unit_tests_dispatch",
        "CommandQueueSingleCardTraceFixture.TensixTestSubDeviceTraceBasicPrograms",
    )
