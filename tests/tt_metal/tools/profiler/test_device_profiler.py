# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os, sys
import json
import re
import inspect
import pytest
import subprocess
import ast
from loguru import logger
from conftest import is_6u

import pandas as pd
import numpy as np

from tracy.common import (
    TT_METAL_HOME,
    PROFILER_HOST_DEVICE_SYNC_INFO,
    PROFILER_SCRIPTS_ROOT,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_LOGS_DIR,
    clear_profiler_runtime_artifacts,
)

from models.common.utility_functions import skip_for_blackhole

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
        "enable_noc_tracing": "TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 ",
        "doDeviceTrace": "TT_METAL_TRACE_PROFILER=1 ",
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
    doDeviceTrace=False,
    slowDispatch=False,
    doSync=False,
    doDispatchCores=False,
):
    name = inspect.stack()[1].function
    testCommand = f"build/{PROG_EXMP_DIR}/{name}"
    if testName:
        testCommand = testName
    clear_profiler_runtime_artifacts()
    envVars = set_env_vars(
        doDeviceTrace=doDeviceTrace,
        slowDispatch=slowDispatch,
        doSync=doSync,
        doDispatchCores=doDispatchCores,
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


def wildcard_match(pattern, words):
    if not pattern.endswith("*"):
        return [word for word in words if pattern == word]
    else:
        prefix = pattern[:-1]
        return [word for word in words if word.startswith(prefix)]


def verify_stats(devicesData, statTypes, allowedRange, refCountDict):
    verifiedStat = []
    for _, deviceData in devicesData["data"]["devices"].items():
        for ref, counts in refCountDict.items():
            matching_refs = wildcard_match(ref, deviceData["cores"]["DEVICE"]["analysis"].keys())
            if matching_refs:
                readCount = 0
                for matching_ref in matching_refs:
                    verifiedStat.append(matching_ref)
                    res = False
                    readCount += deviceData["cores"]["DEVICE"]["analysis"][matching_ref]["stats"]["Count"]
                for count in counts:
                    if count - allowedRange <= readCount <= count + allowedRange:
                        res = True
                        break
                assert (
                    res
                ), f"Wrong tensix zone count for {ref}, read {readCount} which is not within {allowedRange} cycle counts of any of the limits {counts}"

    statTypesSet = set(statTypes)
    for statType in statTypes:
        for stat in verifiedStat:
            if statType in stat and statType in statTypesSet:
                statTypesSet.remove(statType)
    assert (
        len(statTypesSet) == 0
    ), f"Not all required stats (i.e. {statTypesSet}) were found in the device stats (i.e. {verifiedStat})"


def verify_trace_markers(devicesData, num_non_trace_ops, num_trace_ops, num_repeats_per_trace_op):
    for device, deviceData in devicesData["data"]["devices"].items():
        for core, coreData in deviceData["cores"].items():
            for risc, riscData in coreData["riscs"].items():
                non_trace_ops = set()
                trace_ops_to_trace_ids = {}
                trace_ids_to_counts = {}
                for marker in riscData["timeseries"]:
                    marker_data = ast.literal_eval(marker)[0]
                    runtime_id = marker_data["run_host_id"]
                    trace_id = marker_data["trace_id"]
                    if trace_id == -1:
                        non_trace_ops.add(runtime_id)
                    else:
                        if runtime_id not in trace_ops_to_trace_ids:
                            trace_ops_to_trace_ids[runtime_id] = trace_id
                        else:
                            assert (
                                trace_ops_to_trace_ids[runtime_id] == trace_id
                            ), f"Detected multiple trace ids for runtime id {runtime_id}"

                        if trace_id not in trace_ids_to_counts:
                            trace_ids_to_counts[trace_id] = set()
                        trace_ids_to_counts[trace_id].add(int(marker_data["trace_id_count"]))

                # The ops that are being traced may not run on every core on the device. If we detect a core
                # that only runs the first two non-trace ops, we skip it
                if len(non_trace_ops) == 2 and len(trace_ops_to_trace_ids) == 0:
                    continue

                assert (
                    len(non_trace_ops) <= num_non_trace_ops
                ), f"Wrong number of non-trace ops for device {device}, core {core}, risc {risc} - expected at most {num_non_trace_ops}, read {len(non_trace_ops)}"
                assert (
                    len(trace_ops_to_trace_ids) == num_trace_ops
                ), f"Wrong number of trace ops for device {device}, core {core}, risc {risc} - expected {num_trace_ops}, read {len(trace_ops_to_trace_ids)}"

                for trace_id, trace_id_counts in trace_ids_to_counts.items():
                    assert (
                        len(trace_id_counts) == num_repeats_per_trace_op
                    ), f"Wrong number of trace repeats for device {device}, core {core}, risc {risc}, trace {trace_id} - expected {num_repeats_per_trace_op}, read {len(trace_id_counts)}"
                    assert (
                        max(trace_id_counts) == num_repeats_per_trace_op
                    ), f"Wrong maximum trace id counter value for device {device}, core {core}, risc {risc}, trace {trace_id} - expected {num_repeats_per_trace_op}, read {max(trace_id_counts)}"
                    assert (
                        min(trace_id_counts) == 1
                    ), f"Wrong minimum trace id counter value for device {device}, core {core}, risc {risc}, trace {trace_id} - expected 1, read {min(trace_id_counts)}"


def test_trace_run():
    verify_trace_markers(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_trace_runs.py::test_with_ops_multiple_trace_ids"
        ),
        num_non_trace_ops=3,
        num_trace_ops=5,
        num_repeats_per_trace_op=3,
    )

    verify_trace_markers(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_trace_runs.py::test_with_ops_trace_with_non_trace"
        ),
        num_non_trace_ops=12,
        num_trace_ops=10,
        num_repeats_per_trace_op=2,
    )


def test_device_trace_run():
    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_trace_runs.py::test_with_ops",
            setupAutoExtract=False,
            doDeviceTrace=True,
        ),
        statTypes=["kernel", "fw"],
        allowedRange=0,
        refCountDict={
            "trace_fw_duration": [5],
            "trace_kernel_duration": [5],
        },
    )
    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_trace_runs.py::test_with_ops_single_core",
            setupAutoExtract=False,
            doDeviceTrace=True,
        ),
        statTypes=["kernel", "fw"],
        allowedRange=0,
        refCountDict={
            "trace_fw_duration": [5],
            "trace_kernel_duration": [5],
        },
    )


@skip_for_blackhole()
def test_dispatch_cores():
    REF_COUNT_DICT = {
        "Tensix CQ Dispatch*": [600, 760, 1310, 2330],
        "Tensix CQ Prefetch": [900, 1440, 3870, 5000],
        "dispatch_total_cq_cmd_op_time": [236],
        "dispatch_go_send_wait_time": [236],
    }

    verify_stats(
        run_device_profiler_test(setupAutoExtract=True, doDispatchCores=True),
        statTypes=["Dispatch", "Prefetch"],
        allowedRange=150,
        refCountDict=REF_COUNT_DICT,
    )

    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_with_ops -k DispatchCoreType.WORKER",
            setupAutoExtract=True,
            doDispatchCores=True,
        ),
        statTypes=["Dispatch", "Prefetch"],
        allowedRange=1000,
        refCountDict=REF_COUNT_DICT,
    )

    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_all_devices -k DispatchCoreType.WORKER",
            setupAutoExtract=True,
            doDispatchCores=True,
        ),
        statTypes=["Dispatch", "Prefetch"],
        allowedRange=1000,
        refCountDict=REF_COUNT_DICT,
    )

    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_trace_runs.py",
            setupAutoExtract=False,
            doDispatchCores=True,
        ),
        statTypes=["dispatch_total_cq_cmd_op_time", "dispatch_go_send_wait_time"],
        allowedRange=0,  # This test is basically counting ops and should be exact regardless of changes to dispatch code or harvesting.
        refCountDict=REF_COUNT_DICT,
    )


# Eth dispatch will be deprecated
@skip_for_blackhole()
def test_ethernet_dispatch_cores():
    REF_COUNT_DICT = {"Ethernet CQ Dispatch": [590, 840, 1430, 1660, 2320], "Ethernet CQ Prefetch": [572, 4030]}
    devicesData = run_device_profiler_test(
        testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_with_ops -k DispatchCoreType.ETH",
        setupAutoExtract=True,
        doDispatchCores=True,
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
        testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_all_devices -k DispatchCoreType.ETH",
        setupAutoExtract=True,
        doDispatchCores=True,
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
    testCommandArgs = "-tx 3 -ty 3 -sx 2 -sy 2 -rx 1 -ry 1 -m 6 -link -profread"
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
    nocEventsRptPathEnv = f"TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH={PROFILER_ARTIFACTS_DIR}/noc_events_rpt"
    nocEventProfilerEnv = "TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1"
    profilerRun = os.system(f"cd {TT_METAL_HOME} && {nocEventsRptPathEnv} {nocEventProfilerEnv} {testCommand}")
    assert profilerRun == 0

    expected_trace_file = f"{PROFILER_ARTIFACTS_DIR}/noc_events_rpt/noc_trace_dev0_ID0.json"
    assert os.path.isfile(expected_trace_file)

    with open(expected_trace_file, "r") as nocTraceJson:
        noc_trace_data = json.load(nocTraceJson)
        assert len(noc_trace_data) == 8


@skip_for_blackhole()
def test_fabric_event_profiler_1d():
    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in ["wormhole_b0", "blackhole"]

    # test that current device has a valid fabric API connection
    sanity_check_test_bin = "build/test/tt_metal/tt_fabric/fabric_unit_tests"
    sanity_check_test_name = "Fabric1DFixture.TestChipMCast1DWithTracing2"
    sanity_check_succeeded = run_gtest_profiler_test(
        sanity_check_test_bin, sanity_check_test_name, skip_get_device_data=True
    )
    if not sanity_check_succeeded:
        logger.info("Device does not have testable fabric connections, skipping ...")
        return

    # if device supports fabric API, test fabric event profiler
    test_bin = "build/test/tt_metal/tt_fabric/fabric_unit_tests"
    tests = [
        "Fabric1DFixture.TestUnicastRaw",
        "Fabric1DFixture.TestChipMCast1DWithTracing",
        "Fabric1DFixture.TestChipMCast1DWithTracing2",
    ]
    all_tests_expected_event_counts = [
        {frozenset({"start_distance": 1, "range": 1}.items()): 10},
        {frozenset({"start_distance": 1, "range": 3}.items()): 100},
        {frozenset({"start_distance": 2, "range": 2}.items()): 100},
    ]

    for test_name, expected_event_counts in zip(tests, all_tests_expected_event_counts):
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

        noc_trace_files = []
        for f in os.listdir(f"{PROFILER_LOGS_DIR}"):
            if re.match(r"^noc_trace_dev[0-9]+_ID[0-9]+.json$", f):
                noc_trace_files.append(f)

        actual_event_counts = {}
        for trace_file in noc_trace_files:
            with open(f"{PROFILER_LOGS_DIR}/{trace_file}", "r") as nocTraceJson:
                try:
                    noc_trace_data = json.load(nocTraceJson)
                except json.JSONDecodeError:
                    raise ValueError(f"noc trace file '{trace_file}' is not a valid JSON file")

                assert isinstance(noc_trace_data, list), f"noc trace file '{trace_file}' format is incorrect"
                assert len(noc_trace_data) > 0, f"noc trace file '{trace_file}' is empty"
                for event in noc_trace_data:
                    assert isinstance(event, dict), f"noc trace file format error; found event that is not a dict"
                    if event.get("type", "").startswith("FABRIC_"):
                        assert event.get("fabric_send", None) is not None
                        fabric_send_metadata = event.get("fabric_send", None)
                        assert fabric_send_metadata.get("eth_chan", None) is not None
                        del fabric_send_metadata["eth_chan"]
                        key = frozenset(fabric_send_metadata.items())
                        if key not in actual_event_counts:
                            actual_event_counts[key] = 0
                        actual_event_counts[key] += 1

        # compare expected event counts to actual event_counts
        for event in expected_event_counts.keys() | actual_event_counts.keys():
            assert expected_event_counts.get(event, 0) == actual_event_counts.get(
                event, 0
            ), f"There are {actual_event_counts.get(event, 0)} fabric events with fields {event}, expected {expected_event_counts.get(event, 0)}"


@skip_for_blackhole()
def test_fabric_event_profiler_fabric_mux():
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
    tests = ["Fabric1DMuxFixture.TestFabricMuxTwoChipVariantWithNocTracing"]
    expected_outputs = [
        {"FABRIC_EVENT_COUNT": 400},
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

        noc_trace_files = []
        for f in os.listdir(f"{PROFILER_LOGS_DIR}"):
            if re.match(r"^noc_trace_dev[0-9]+_ID[0-9]+.json$", f):
                noc_trace_files.append(f)

        fabric_event_count = 0
        for trace_file in noc_trace_files:
            with open(f"{PROFILER_LOGS_DIR}/{trace_file}", "r") as nocTraceJson:
                try:
                    noc_trace_data = json.load(nocTraceJson)
                except json.JSONDecodeError:
                    raise ValueError(f"noc trace file '{trace_file}' is not a valid JSON file")

                assert isinstance(noc_trace_data, list), f"noc trace file '{trace_file}' format is incorrect"
                assert len(noc_trace_data) > 0, f"noc trace file '{trace_file}' is empty"
                for event in noc_trace_data:
                    assert isinstance(event, dict), f"noc trace file format error; found event that is not a dict"
                    if event.get("type", "").startswith("FABRIC_"):
                        fabric_event_count += 1
                        assert event.get("fabric_send", None) is not None

                        fabric_send_metadata = event.get("fabric_send", None)
                        assert fabric_send_metadata.get("eth_chan", None) is not None
                        assert fabric_send_metadata.get("start_distance", None) is not None
                        assert fabric_send_metadata.get("range", None) is not None
                        assert fabric_send_metadata.get("fabric_mux", None) is not None

                        fabric_mux_metadata = fabric_send_metadata.get("fabric_mux", None)
                        assert fabric_mux_metadata.get("x", None) is not None
                        assert fabric_mux_metadata.get("y", None) is not None
                        assert fabric_mux_metadata.get("noc", None) is not None

        assert (
            fabric_event_count == expected_output["FABRIC_EVENT_COUNT"]
        ), f"Incorrect number of fabric events found in noc trace: {fabric_event_count}, expected {expected_output['FABRIC_EVENT_COUNT']}"


@skip_for_blackhole()
def test_fabric_event_profiler_2d():
    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in ["wormhole_b0", "blackhole"]

    # test that current device has a valid fabric API connection
    sanity_check_test_bin = "build/test/tt_metal/tt_fabric/fabric_unit_tests"
    sanity_check_test_name = "Fabric2DFixture.Test2DMCastConnAPI_1N1E1W"
    sanity_check_succeeded = run_gtest_profiler_test(
        sanity_check_test_bin, sanity_check_test_name, skip_get_device_data=True
    )
    if not sanity_check_succeeded:
        logger.info("Device does not have testable fabric connections, skipping ...")
        return

    # if device supports fabric API, test fabric event profiler
    test_bin = "build/test/tt_metal/tt_fabric/fabric_unit_tests"
    tests = [
        "Fabric2DFixture.TestUnicastRaw_3E",
        "Fabric2DFixture.TestMCastConnAPI_1W2E",
        "Fabric2DFixture.Test2DMCastConnAPI_1N1E1W",
    ]

    if is_6u():
        tests.extend(
            [
                "Fabric2DFixture.TestUnicastRaw_3N",
                "Fabric2DFixture.TestUnicastRaw_3N3E",
                "Fabric2DFixture.TestMCastConnAPI_2N1S",
                "Fabric2DFixture.Test2DMCastConnAPI_7N3E",
            ]
        )

    all_tests_expected_event_counts = [
        {
            frozenset({"ns_hops": 0, "e_hops": 3, "w_hops": 0, "is_mcast": False}.items()): 10,
        },
        {
            frozenset({"ns_hops": 0, "e_hops": 0, "w_hops": 1, "is_mcast": False}.items()): 100,
            frozenset({"ns_hops": 0, "e_hops": 2, "w_hops": 0, "is_mcast": True}.items()): 100,
        },
        {
            frozenset({"ns_hops": 1, "e_hops": 1, "w_hops": 1, "is_mcast": True}.items()): 100,
            frozenset({"ns_hops": 0, "e_hops": 1, "w_hops": 0, "is_mcast": False}.items()): 100,
            frozenset({"ns_hops": 0, "e_hops": 0, "w_hops": 1, "is_mcast": False}.items()): 100,
        },
    ]

    if is_6u():
        all_tests_expected_event_counts.extend(
            [
                {
                    frozenset({"ns_hops": 3, "e_hops": 0, "w_hops": 0, "is_mcast": False}.items()): 10,
                },
                {
                    frozenset({"ns_hops": 2, "e_hops": 4, "w_hops": 0, "is_mcast": False}.items()): 10,
                },
                {
                    frozenset({"ns_hops": 2, "e_hops": 0, "w_hops": 0, "is_mcast": True}.items()): 100,
                    frozenset({"ns_hops": 1, "e_hops": 0, "w_hops": 0, "is_mcast": False}.items()): 100,
                },
                {
                    frozenset({"ns_hops": 7, "e_hops": 3, "w_hops": 0, "is_mcast": True}.items()): 100,
                    frozenset({"ns_hops": 0, "e_hops": 3, "w_hops": 0, "is_mcast": True}.items()): 100,
                },
            ]
        )

    for test_name, expected_event_counts in zip(tests, all_tests_expected_event_counts):
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

        noc_trace_files = []
        for f in os.listdir(f"{PROFILER_LOGS_DIR}"):
            if re.match(r"^noc_trace_dev[0-9]+_ID[0-9]+.json$", f):
                noc_trace_files.append(f)

        actual_event_counts = {}
        for trace_file in noc_trace_files:
            with open(f"{PROFILER_LOGS_DIR}/{trace_file}", "r") as nocTraceJson:
                try:
                    noc_trace_data = json.load(nocTraceJson)
                except json.JSONDecodeError:
                    raise ValueError(f"noc trace file '{trace_file}' is not a valid JSON file")

                assert isinstance(noc_trace_data, list), f"noc trace file '{trace_file}' format is incorrect"
                assert len(noc_trace_data) > 0, f"noc trace file '{trace_file}' is empty"
                for event in noc_trace_data:
                    assert isinstance(event, dict), f"noc trace file format error; found event that is not a dict"
                    if event.get("type", "").startswith("FABRIC_"):
                        assert event.get("fabric_send", None) is not None
                        fabric_send_metadata = event.get("fabric_send", None)
                        assert fabric_send_metadata.get("eth_chan", None) is not None
                        del fabric_send_metadata["eth_chan"]
                        key = frozenset(fabric_send_metadata.items())
                        if key not in actual_event_counts:
                            actual_event_counts[key] = 0
                        actual_event_counts[key] += 1

        # compare expected event counts to actual event_counts
        for event in expected_event_counts.keys() | actual_event_counts.keys():
            assert expected_event_counts.get(event, 0) == actual_event_counts.get(
                event, 0
            ), f"There are {actual_event_counts.get(event, 0)} fabric events with fields {event}, expected {expected_event_counts.get(event, 0)}"


def test_sub_device_profiler():
    ARCH_NAME = os.getenv("ARCH_NAME")
    run_gtest_profiler_test(
        "./build/test/tt_metal/unit_tests_dispatch",
        "UnitMeshCQSingleCardFixture.TensixTestSubDeviceBasicPrograms",
    )
    run_gtest_profiler_test(
        "./build/test/tt_metal/unit_tests_dispatch",
        "UnitMeshCQSingleCardTraceFixture.TensixTestSubDeviceTraceBasicPrograms",
    )
