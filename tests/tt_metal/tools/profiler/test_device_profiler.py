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
import multiprocessing as mp

from tracy.common import (
    TT_METAL_HOME,
    PROFILER_HOST_DEVICE_SYNC_INFO,
    PROFILER_SCRIPTS_ROOT,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_LOGS_DIR,
    PROFILER_CPP_DEVICE_PERF_REPORT,
    PROFILER_DEFAULT_OP_SUPPORT_COUNT,
    clear_profiler_runtime_artifacts,
)

from models.common.utility_functions import skip_for_blackhole

PROG_EXMP_DIR = "programming_examples/profiler"
TRACY_TESTS_DIR = "./tests/ttnn/tracy"


def is_6u_wrapper():
    ctx = mp.get_context("spawn")
    with ctx.Pool() as pool:
        result = pool.apply(is_6u)
        pool.close()
        pool.join()
    return result


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
        "do_mid_run_dump": "TT_METAL_PROFILER_MID_RUN_DUMP=1 ",
        "do_cpp_post_process": "TT_METAL_PROFILER_CPP_POST_PROCESS=1 ",
        "set_program_support_count": "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=",
    }
    envVarsStr = " "
    for arg, argVal in kwargs.items():
        if arg == "set_program_support_count":
            # Only set the program support count here if it's not equal to the default program support count and the environment variable isn't already set
            if (
                argVal
                and argVal != PROFILER_DEFAULT_OP_SUPPORT_COUNT
                and os.getenv("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT") is None
            ):
                envVarsStr += f"{envVarsDict[arg]}{argVal} "
        else:
            if argVal:
                envVarsStr += f"{envVarsDict[arg]}"
    return envVarsStr


# returns True if test passed, False if test was SKIPPED
def run_gtest_profiler_test(
    testbin,
    testname,
    doSync=False,
    enable_noc_tracing=False,
    do_mid_run_dump=False,
    do_cpp_post_process=False,
    skip_get_device_data=False,
):
    clear_profiler_runtime_artifacts()
    envVars = set_env_vars(
        doSync=doSync,
        enable_noc_tracing=enable_noc_tracing,
        do_mid_run_dump=do_mid_run_dump,
        do_cpp_post_process=do_cpp_post_process,
    )
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
    noPostProcess=False,
    setupAutoExtract=False,
    doDeviceTrace=False,
    slowDispatch=False,
    doSync=False,
    enable_noc_tracing=False,
    doDispatchCores=False,
    setOpSupportCount=PROFILER_DEFAULT_OP_SUPPORT_COUNT,
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
        enable_noc_tracing=enable_noc_tracing,
        doDispatchCores=doDispatchCores,
        set_program_support_count=setOpSupportCount,
    )
    testCommand = f"cd {TT_METAL_HOME} && {envVars} {testCommand}"
    print()
    logger.info(f"Running: {testCommand}")
    profilerRun = os.system(testCommand)
    assert profilerRun == 0

    if noPostProcess:
        return None

    setupStr = ""
    if setupAutoExtract:
        setupStr = f"-s {name}"

    return get_device_data(setupStr)


def get_function_name():
    frame = inspect.currentframe()
    return frame.f_code.co_name


@pytest.mark.skip_post_commit
def test_multi_op():
    OP_COUNT = 1000
    RUN_COUNT = 2
    REF_COUNT_DICT = {
        "wormhole_b0": [72 * OP_COUNT * RUN_COUNT, 64 * OP_COUNT * RUN_COUNT, 56 * OP_COUNT * RUN_COUNT],
        "blackhole": [130 * OP_COUNT * RUN_COUNT, 120 * OP_COUNT * RUN_COUNT, 110 * OP_COUNT * RUN_COUNT],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setupAutoExtract=True, setOpSupportCount=1200)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    statName = f"BRISC KERNEL_START->KERNEL_END"

    assert statName in stats.keys(), "Wrong device analysis format"
    assert stats[statName]["stats"]["Count"] in REF_COUNT_DICT[ENV_VAR_ARCH_NAME], "Wrong Marker Repeat count"


@pytest.mark.skip_post_commit
def test_multi_op_buffer_overflow():
    COMPUTE_OP_COUNT = 200
    DATA_MOVEMENT_OP_COUNT = 1000
    RUN_COUNT = 1
    REF_COMPUTE_COUNT_DICT = {
        "wormhole_b0": [
            72 * COMPUTE_OP_COUNT * RUN_COUNT,
            64 * COMPUTE_OP_COUNT * RUN_COUNT,
            56 * COMPUTE_OP_COUNT * RUN_COUNT,
        ],
        "blackhole": [
            130 * COMPUTE_OP_COUNT * RUN_COUNT,
            120 * COMPUTE_OP_COUNT * RUN_COUNT,
            110 * COMPUTE_OP_COUNT * RUN_COUNT,
        ],
    }
    REF_DATA_MOVEMENT_COUNT_DICT = {
        "wormhole_b0": [
            72 * DATA_MOVEMENT_OP_COUNT * RUN_COUNT,
            64 * DATA_MOVEMENT_OP_COUNT * RUN_COUNT,
            56 * DATA_MOVEMENT_OP_COUNT * RUN_COUNT,
        ],
        "blackhole": [
            130 * DATA_MOVEMENT_OP_COUNT * RUN_COUNT,
            120 * DATA_MOVEMENT_OP_COUNT * RUN_COUNT,
            110 * DATA_MOVEMENT_OP_COUNT * RUN_COUNT,
        ],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COMPUTE_COUNT_DICT.keys()
    assert ENV_VAR_ARCH_NAME in REF_DATA_MOVEMENT_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setupAutoExtract=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    for risc in ["BRISC", "NCRISC"]:
        statName = f"{risc} KERNEL_START->KERNEL_END"

        assert statName in stats.keys(), "Wrong device analysis format"
        assert (
            stats[statName]["stats"]["Count"] in REF_DATA_MOVEMENT_COUNT_DICT[ENV_VAR_ARCH_NAME]
        ), "Wrong Marker Repeat count"

    for risc in ["TRISC_0", "TRISC_1", "TRISC_2"]:
        statName = f"{risc} KERNEL_START->KERNEL_END"

        assert statName in stats.keys(), "Wrong device analysis format"
        assert (
            stats[statName]["stats"]["Count"] in REF_COMPUTE_COUNT_DICT[ENV_VAR_ARCH_NAME]
        ), "Wrong Marker Repeat count"


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


@pytest.mark.skip_post_commit
def test_full_buffer():
    OP_COUNT = 23
    RISC_COUNT = 5
    ZONE_COUNT = 125
    REF_COUNT_DICT = {
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


@pytest.mark.skip_post_commit
def test_device_api_debugger_non_dropping():
    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in ["grayskull", "wormhole_b0", "blackhole"]

    testCommand = f"build/{PROG_EXMP_DIR}/test_device_api_debugger"
    clear_profiler_runtime_artifacts()

    envVars = "TT_METAL_DEVICE_DEBUG_DUMP_ENABLED=1 "

    profilerRun = os.system(f"cd {TT_METAL_HOME} && {envVars} {testCommand}")
    assert profilerRun == 0, f"Test command failed with exit code {profilerRun}"

    # Verify the NOC trace JSON file exists
    expected_trace_file = f"{PROFILER_LOGS_DIR}/noc_trace_dev0_ID0.json"
    assert os.path.isfile(expected_trace_file), f"Expected trace file '{expected_trace_file}' does not exist"

    # Read and parse the JSON file
    with open(expected_trace_file, "r") as nocTraceJson:
        try:
            noc_trace_data = json.load(nocTraceJson)
        except json.JSONDecodeError:
            raise ValueError(f"noc trace file '{expected_trace_file}' is not a valid JSON file")

    assert isinstance(noc_trace_data, list), f"noc trace file '{expected_trace_file}' format is incorrect"
    assert len(noc_trace_data) > 0, f"noc trace file '{expected_trace_file}' is empty"

    # Track READ and WRITE events separately for each RISC
    brisc_read_dst_addrs_found = set()
    brisc_write_dst_addrs_found = set()
    ncrisc_read_dst_addrs_found = set()
    ncrisc_write_dst_addrs_found = set()

    # Track barrier events
    read_barrier_start_count = 0
    read_barrier_end_count = 0
    write_barrier_start_count = 0
    write_barrier_end_count = 0

    for event in noc_trace_data:
        assert isinstance(event, dict), f"noc trace file format error; found event that is not a dict"
        event_type = event.get("type")

        # Count barrier events
        if event_type == "READ_BARRIER_START":
            read_barrier_start_count += 1
        elif event_type == "READ_BARRIER_END":
            read_barrier_end_count += 1
        elif event_type == "WRITE_BARRIER_START":
            write_barrier_start_count += 1
        elif event_type == "WRITE_BARRIER_END":
            write_barrier_end_count += 1

        if "dst_addr" in event and "proc" in event:
            proc = event["proc"]
            dst_addr = event["dst_addr"]

            if proc == "BRISC":
                if event_type == "READ":
                    brisc_read_dst_addrs_found.add(dst_addr)
                elif event_type == "WRITE_":
                    brisc_write_dst_addrs_found.add(dst_addr)
            elif proc == "NCRISC":
                if event_type == "READ":
                    ncrisc_read_dst_addrs_found.add(dst_addr)
                elif event_type == "WRITE_":
                    ncrisc_write_dst_addrs_found.add(dst_addr)

    expected_dst_addrs = set(range(10000))  # 0 to 9999

    # Verify BRISC READ has all expected dst_addr values
    missing_brisc_read_dst_addrs = expected_dst_addrs - brisc_read_dst_addrs_found
    assert len(missing_brisc_read_dst_addrs) == 0, (
        f"Missing dst_addr values in BRISC READ JSON events: {sorted(missing_brisc_read_dst_addrs)[:20]}"
        f"{'...' if len(missing_brisc_read_dst_addrs) > 20 else ''} "
        f"(found {len(brisc_read_dst_addrs_found)} out of {len(expected_dst_addrs)} expected)"
    )

    # Verify BRISC WRITE has all expected dst_addr values
    missing_brisc_write_dst_addrs = expected_dst_addrs - brisc_write_dst_addrs_found
    assert len(missing_brisc_write_dst_addrs) == 0, (
        f"Missing dst_addr values in BRISC WRITE JSON events: {sorted(missing_brisc_write_dst_addrs)[:20]}"
        f"{'...' if len(missing_brisc_write_dst_addrs) > 20 else ''} "
        f"(found {len(brisc_write_dst_addrs_found)} out of {len(expected_dst_addrs)} expected)"
    )

    # Verify NCRISC READ has all expected dst_addr values
    missing_ncrisc_read_dst_addrs = expected_dst_addrs - ncrisc_read_dst_addrs_found
    assert len(missing_ncrisc_read_dst_addrs) == 0, (
        f"Missing dst_addr values in NCRISC READ JSON events: {sorted(missing_ncrisc_read_dst_addrs)[:20]}"
        f"{'...' if len(missing_ncrisc_read_dst_addrs) > 20 else ''} "
        f"(found {len(ncrisc_read_dst_addrs_found)} out of {len(expected_dst_addrs)} expected)"
    )

    # Verify NCRISC WRITE has all expected dst_addr values
    missing_ncrisc_write_dst_addrs = expected_dst_addrs - ncrisc_write_dst_addrs_found
    assert len(missing_ncrisc_write_dst_addrs) == 0, (
        f"Missing dst_addr values in NCRISC WRITE JSON events: {sorted(missing_ncrisc_write_dst_addrs)[:20]}"
        f"{'...' if len(missing_ncrisc_write_dst_addrs) > 20 else ''} "
        f"(found {len(ncrisc_write_dst_addrs_found)} out of {len(expected_dst_addrs)} expected)"
    )

    # Verify barrier event counts
    expected_barrier_count = 20000
    assert (
        read_barrier_start_count == expected_barrier_count
    ), f"Expected {expected_barrier_count} READ_BARRIER_START events, found {read_barrier_start_count}"
    assert (
        read_barrier_end_count == expected_barrier_count
    ), f"Expected {expected_barrier_count} READ_BARRIER_END events, found {read_barrier_end_count}"
    assert (
        write_barrier_start_count == expected_barrier_count
    ), f"Expected {expected_barrier_count} WRITE_BARRIER_START events, found {write_barrier_start_count}"
    assert (
        write_barrier_end_count == expected_barrier_count
    ), f"Expected {expected_barrier_count} WRITE_BARRIER_END events, found {write_barrier_end_count}"


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
    # Some traced ops may not run on every core/risc. We therefore:
    # - validate trace-id / trace-id-counter consistency for any markers we do observe per core/risc
    # - validate the total number of unique traced ops at the device level
    device_to_all_trace_runtime_ids = {}

    for device, deviceData in devicesData["data"]["devices"].items():
        device_to_all_trace_runtime_ids.setdefault(device, set())
        for core, coreData in deviceData["cores"].items():
            for risc, riscData in coreData["riscs"].items():
                non_trace_ops = set()
                trace_ops_to_trace_ids = {}
                trace_ids_to_counts = {}
                for marker in riscData["timeseries"]:
                    marker_data = ast.literal_eval(marker)[0] if isinstance(marker, str) else marker[0]
                    runtime_id = marker_data["run_host_id"]
                    trace_id = marker_data["trace_id"]
                    if trace_id == -1:
                        non_trace_ops.add(runtime_id)
                    else:
                        device_to_all_trace_runtime_ids[device].add(runtime_id)
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
                    len(trace_ops_to_trace_ids) <= num_trace_ops
                ), f"Wrong number of trace ops for device {device}, core {core}, risc {risc} - expected at most {num_trace_ops}, read {len(trace_ops_to_trace_ids)}"

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

    # Validate that the traced workload actually generated the expected number of unique traced ops.
    for device, runtime_ids in device_to_all_trace_runtime_ids.items():
        assert (
            len(runtime_ids) == num_trace_ops
        ), f"Wrong total number of unique trace ops for device {device} - expected {num_trace_ops}, read {len(runtime_ids)}"


def verify_trace_replay_ids_match_between_riscs(devicesData, riscs=("BRISC", "NCRISC")):
    """
    Verify that for *trace replay* ops, both riscs report consistent TRACE ID and TRACE REPLAY ID.

    Concretely, we use the kernel main() zone markers (e.g. BRISC-KERNEL / NCRISC-KERNEL) as a stable
    per-op signal, and validate that:
      - Both riscs emit replay-tagged kernel markers (trace_id != -1 and trace_id_count != -1)
      - For ops observed on *both* riscs, the (trace_id, trace_id_count) pairs match
      - For ops observed on only one risc (common when one risc doesn't participate), the trace_id used
        is consistent with the trace_id used by the other risc on that core.
    """

    def _parse_timer_id(timeseries_entry):
        # JSON encoder stores tuples as strings, so parse when needed.
        if isinstance(timeseries_entry, str):
            return ast.literal_eval(timeseries_entry)[0]
        return timeseries_entry[0]

    def _collect_replay_kernel_starts(risc_timeseries, expected_zone_name):
        # Map run_host_id -> { trace_id_count -> trace_id } for replay kernel-start markers.
        out = {}
        for entry in risc_timeseries:
            timer_id = _parse_timer_id(entry)
            if timer_id.get("type") != "ZONE_START":
                continue
            if timer_id.get("zone_name") != expected_zone_name:
                continue
            trace_id = int(timer_id.get("trace_id", -1))
            trace_id_count = int(timer_id.get("trace_id_count", -1))
            if trace_id == -1 or trace_id_count == -1:
                # Not a trace replay marker
                continue
            run_host_id = int(timer_id["run_host_id"])
            if run_host_id not in out:
                out[run_host_id] = {}
            if trace_id_count in out[run_host_id]:
                assert (
                    out[run_host_id][trace_id_count] == trace_id
                ), f"Detected multiple trace ids for replay (run_host_id={run_host_id}, trace_id_count={trace_id_count})"
            else:
                out[run_host_id][trace_id_count] = trace_id
        return out

    zone_names = {
        "BRISC": "BRISC-KERNEL",
        "NCRISC": "NCRISC-KERNEL",
    }
    for risc in riscs:
        assert risc in zone_names, f"Unsupported risc '{risc}' for trace replay verification"

    checked_any_common_op = False
    for device, deviceData in devicesData["data"]["devices"].items():
        for core, coreData in deviceData["cores"].items():
            # Skip the synthetic aggregate entry.
            if core == "DEVICE":
                continue
            core_riscs = coreData.get("riscs", {})
            if any(risc not in core_riscs for risc in riscs):
                continue

            per_risc_maps = {}
            for risc in riscs:
                per_risc_maps[risc] = _collect_replay_kernel_starts(core_riscs[risc]["timeseries"], zone_names[risc])

            # Only validate ops that are present in *both* riscs. BRISC can run out of buffer/DRAM space
            # earlier than other riscs, so requiring full coverage would be flaky.
            common_run_host_ids = set.intersection(*(set(per_risc_maps[r].keys()) for r in riscs))
            if len(common_run_host_ids) == 0:
                continue
            checked_any_common_op = True

            # Derive an expected trace_id set from the first risc on this core.
            # (These tests execute a single trace id, so this should be a singleton set.)
            ref_risc = riscs[0]
            expected_trace_ids = {
                trace_id for per_run in per_risc_maps[ref_risc].values() for trace_id in per_run.values()
            }

            # All riscs should use only the expected trace ids for replay-tagged kernel markers.
            for risc in riscs[1:]:
                risc_trace_ids = {trace_id for per_run in per_risc_maps[risc].values() for trace_id in per_run.values()}
                unexpected = sorted(list(risc_trace_ids - expected_trace_ids))[:25]
                assert (
                    len(unexpected) == 0
                ), f"Unexpected trace_id values for device {device}, core {core}, risc {risc}: {unexpected}. Expected subset of {sorted(list(expected_trace_ids))}"

            # For ops that appear on multiple riscs, validate that each risc reports a replay id >= 1
            # (i.e. trace_id_count is present and positive). We avoid stricter equality constraints
            # because one risc may miss some replays due to log buffer pressure.
            for run_host_id in common_run_host_ids:
                for risc in riscs[1:]:
                    counts = set(per_risc_maps[risc][run_host_id].keys())
                    assert len(counts) > 0, (
                        f"Missing trace replay session ids for device {device}, core {core}, risc {risc}, "
                        f"run_host_id {run_host_id}"
                    )
                    assert (
                        min(counts) >= 1
                    ), f"Invalid trace replay session id for device {device}, core {core}, risc {risc}, run_host_id {run_host_id}: min(trace_id_count)={min(counts)}"

                ref_counts = set(per_risc_maps[ref_risc][run_host_id].keys())
                assert len(ref_counts) > 0, (
                    f"Missing trace replay session ids for device {device}, core {core}, risc {ref_risc}, "
                    f"run_host_id {run_host_id}"
                )
                assert (
                    min(ref_counts) >= 1
                ), f"Invalid trace replay session id for device {device}, core {core}, risc {ref_risc}, run_host_id {run_host_id}: min(trace_id_count)={min(ref_counts)}"

    assert (
        checked_any_common_op
    ), "No replay-tagged ops were present in both riscs; test did not exercise cross-risc trace replay logging"


def verify_noc_trace_replay_ids_have_risc_coverage(
    profiler_logs_dir=PROFILER_LOGS_DIR, device_id="0", riscs=("BRISC", "NCRISC")
):
    """
    Verification for the quick-push + NOC-trace path.

    NOC trace output filenames encode trace information via:
        _traceID((trace_id << 32) | trace_id_counter)

    This check ensures that for any replay-tagged op (i.e. has _traceID in filename) where we ever
    observe *both* riscs, we continue to observe *both* riscs for every replay id. This guards against
    the original bug where a non-BRISC core would quick-push NOC events with a stale trace id/counter,
    causing BRISC/NCRISC events for the same replay to be split across different _traceID files.
    """

    # Parse: noc_trace_dev<dev>[_<op_name>]_ID<runtime_id>[_traceID<encoded>].json
    noc_trace_re = re.compile(
        r"^noc_trace_dev(?P<dev>[0-9]+)(?:_.*)?_ID(?P<rid>[0-9]+)(?:_traceID(?P<trace>[0-9]+))?\\.json$"
    )

    per_runtime_per_trace = {}
    for fname in os.listdir(profiler_logs_dir):
        m = noc_trace_re.match(fname)
        if not m:
            continue
        if m.group("dev") != str(device_id):
            continue
        trace_enc = m.group("trace")
        if trace_enc is None:
            continue  # non-trace op

        runtime_id = int(m.group("rid"))
        trace_enc = int(trace_enc)
        trace_id = trace_enc >> 32
        trace_id_counter = trace_enc & 0xFFFFFFFF
        assert trace_id_counter > 0, f"Invalid trace replay counter in NOC trace filename '{fname}'"

        with open(os.path.join(profiler_logs_dir, fname), "r") as f:
            contents = f.read()

        # Avoid JSON parsing; substring presence is enough for coverage checks.
        procs_present = {risc for risc in riscs if f'"proc": "{risc}"' in contents}

        key = (runtime_id, trace_id, trace_id_counter)
        if key in per_runtime_per_trace:
            per_runtime_per_trace[key] |= procs_present
        else:
            per_runtime_per_trace[key] = set(procs_present)

    assert (
        len(per_runtime_per_trace) > 0
    ), f"No replay-tagged NOC trace files found in '{profiler_logs_dir}' for device {device_id}"

    # Determine which runtime_ids are expected to contain both riscs (they do at least once).
    runtime_to_all_procs = {}
    runtime_to_trace_ids = {}
    runtime_to_counters = {}
    for (runtime_id, trace_id, trace_id_counter), procs_present in per_runtime_per_trace.items():
        runtime_to_all_procs.setdefault(runtime_id, set()).update(procs_present)
        runtime_to_trace_ids.setdefault(runtime_id, set()).add(trace_id)
        runtime_to_counters.setdefault(runtime_id, set()).add(trace_id_counter)

    dual_risc_runtime_ids = [rid for rid, procs in runtime_to_all_procs.items() if set(riscs).issubset(procs)]
    assert (
        len(dual_risc_runtime_ids) > 0
    ), f"Did not find any replay-tagged ops that included all riscs {riscs}; cannot validate risc coverage"

    # For those runtime_ids, every replay id we observed must include both riscs.
    for runtime_id in dual_risc_runtime_ids:
        assert (
            len(runtime_to_trace_ids.get(runtime_id, set())) == 1
        ), f"Multiple trace_ids observed for runtime_id {runtime_id}: {sorted(list(runtime_to_trace_ids[runtime_id]))}"
        trace_id = next(iter(runtime_to_trace_ids[runtime_id]))
        for trace_id_counter in sorted(runtime_to_counters.get(runtime_id, set())):
            procs_present = per_runtime_per_trace.get((runtime_id, trace_id, trace_id_counter), set())
            missing = sorted(list(set(riscs) - procs_present))
            assert (
                len(missing) == 0
            ), f"Missing risc coverage in NOC trace for device {device_id}, runtime_id {runtime_id}, trace_id {trace_id}, trace_id_counter {trace_id_counter}. Missing riscs: {missing}"


def test_trace_run():
    verify_trace_markers(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_trace_runs.py::test_with_ops_multiple_trace_ids"
        ),
        num_non_trace_ops=4,
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


@pytest.mark.skip_post_commit
def test_quick_push_on_noc_profiler():
    devicesData = run_device_profiler_test(
        testName=f"pytest {TRACY_TESTS_DIR}/test_trace_runs.py::test_with_ops",
        enable_noc_tracing=True,
    )

    # BRISC can run out of buffer/DRAM space earlier than other riscs, so only validate ops that are
    # present in *both* BRISC and NCRISC logs. Those must have valid trace ids and replay session ids.
    verify_trace_replay_ids_match_between_riscs(devicesData, riscs=("BRISC", "NCRISC"))


@skip_for_blackhole()
def test_dispatch_cores():
    REF_COUNT_DICT = {
        "Tensix CQ Dispatch*": [9325],
        "Tensix CQ Prefetch": [9325],
    }

    verify_stats(
        run_device_profiler_test(setupAutoExtract=True, doDispatchCores=True),
        statTypes=["Dispatch", "Prefetch"],
        allowedRange=8875,
        refCountDict=REF_COUNT_DICT,
    )


@skip_for_blackhole()
@pytest.mark.skip_post_commit
def test_dispatch_cores_extended_worker():
    REF_COUNT_DICT = {
        "Tensix CQ Dispatch*": [9325],
        "Tensix CQ Prefetch": [9325],
        "dispatch_total_cq_cmd_op_time": [87],
        "dispatch_go_send_wait_time": [87],
    }

    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_with_ops -k DispatchCoreType.WORKER",
            setupAutoExtract=True,
            doDispatchCores=True,
            setOpSupportCount=1500,
        ),
        statTypes=["Dispatch", "Prefetch"],
        allowedRange=9260,
        refCountDict=REF_COUNT_DICT,
    )

    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_mesh_device -k DispatchCoreType.WORKER",
            setupAutoExtract=True,
            doDispatchCores=True,
            setOpSupportCount=3000,
        ),
        statTypes=["Dispatch", "Prefetch"],
        allowedRange=9260,
        refCountDict=REF_COUNT_DICT,
    )

    verify_stats(
        run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::test_with_ops -k DispatchCoreType.WORKER",
            setupAutoExtract=False,
            doDispatchCores=True,
        ),
        statTypes=["dispatch_total_cq_cmd_op_time", "dispatch_go_send_wait_time"],
        allowedRange=0,  # This test is basically counting ops and should be exact regardless of changes to dispatch code or harvesting.
        refCountDict=REF_COUNT_DICT,
    )


def _validate_ethernet_dispatch_counts(devicesData, min_count, max_count):
    """
    Helper function to validate ethernet dispatch counts are within expected range.

    Args:
        devicesData: Device data from run_device_profiler_test
        min_count: Minimum acceptable count value
        max_count: Maximum acceptable count value
    """
    stat_names = ["Ethernet CQ Dispatch", "Ethernet CQ Prefetch"]

    for device, deviceData in devicesData["data"]["devices"].items():
        for stat_name in stat_names:
            if stat_name in deviceData["cores"]["DEVICE"]["analysis"].keys():
                read_count = deviceData["cores"]["DEVICE"]["analysis"][stat_name]["stats"]["Count"]
                assert min_count <= read_count <= max_count, (
                    f"Wrong ethernet dispatch count for '{stat_name}' on device {device}: "
                    f"read {read_count}, expected between {min_count} and {max_count}"
                )


# Eth dispatch will be deprecated
@skip_for_blackhole()
@pytest.mark.skip_post_commit
@pytest.mark.skipif(is_6u_wrapper(), reason="Ethernet dispatch is not needed to be tested on 6U")
def test_ethernet_dispatch_cores():
    # Simple range check: both Dispatch and Prefetch should be within this range
    MIN_COUNT = 500
    MAX_COUNT = 10000

    # Test configuration: (test_name_suffix, op_support_count)
    test_configs = [
        ("test_with_ops", 1500),
        ("test_mesh_device", 3000),
    ]

    for test_suffix, op_support_count in test_configs:
        devicesData = run_device_profiler_test(
            testName=f"pytest {TRACY_TESTS_DIR}/test_dispatch_profiler.py::{test_suffix} -k DispatchCoreType.ETH",
            setupAutoExtract=True,
            doDispatchCores=True,
            setOpSupportCount=op_support_count,
        )
        _validate_ethernet_dispatch_counts(devicesData, MIN_COUNT, MAX_COUNT)


def test_profiler_host_device_sync():
    TOLERANCE = 0.1

    syncInfoFile = PROFILER_LOGS_DIR / PROFILER_HOST_DEVICE_SYNC_INFO

    deviceData = run_device_profiler_test(
        testName=f"pytest {TRACY_TESTS_DIR}/test_profiler_sync.py::test_mesh_device", doSync=True
    )
    reportedFreq = deviceData["data"]["deviceInfo"]["freq"] * 1e6
    assert os.path.isfile(syncInfoFile)

    syncinfoDF = pd.read_csv(syncInfoFile)
    devices = sorted(syncinfoDF["device id"].unique())
    available_devices = sorted(int(device_id) for device_id in deviceData["data"]["devices"].keys())
    missing_devices = [device_id for device_id in available_devices if device_id not in devices]
    assert len(missing_devices) == 0, f"Missing sync info for devices {missing_devices}"
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
    available_devices = sorted(int(device_id) for device_id in deviceData["data"]["devices"].keys())
    missing_devices = [device_id for device_id in available_devices if device_id not in devices]
    assert len(missing_devices) == 0, f"Missing sync info for devices {missing_devices}"
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
    WH_ERISC_COUNTS = [0, 3, 6, 16]  # N150, N300, T3K, 6U
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
    is_6u_bool = is_6u_wrapper()

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

    if is_6u_bool:
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

    if is_6u_bool:
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


def validate_programs_perf_durations(perf_data):
    cpp_ops_perf_report = pd.read_csv(PROFILER_LOGS_DIR / PROFILER_CPP_DEVICE_PERF_REPORT).reset_index(drop=True)

    for snapshot in perf_data:
        for device in snapshot:
            device_id = device["device"]
            device_programs_analysis_data = device["programs_analysis_data"]
            for program_analysis_data in device_programs_analysis_data:
                runtime_id = program_analysis_data["program_execution_uid"]["runtime_id"]
                program_analyses_results = program_analysis_data["program_analyses_results"]
                for analysis_type in program_analyses_results:
                    analysis_result = program_analyses_results[analysis_type]
                    if analysis_result["duration"] == 0:
                        continue

                    row = cpp_ops_perf_report.loc[
                        (cpp_ops_perf_report["GLOBAL CALL COUNT"] == runtime_id)
                        & (cpp_ops_perf_report["DEVICE ID"] == device_id)
                    ]

                    assert row[analysis_type].values[0] == analysis_result["duration"]


def test_get_programs_perf_data():
    # Program execution UIDs and the number of programs are validated in the test_get_programs_perf_data gtests
    # In this file, we validate the durations of the programs
    test_get_programs_perf_data_binary = "./build/test/ttnn/tracy/test_get_programs_perf_data"

    run_gtest_profiler_test(
        test_get_programs_perf_data_binary,
        "GetProgramsPerfDataFixture.TestGetProgramsPerfDataBeforeReadMeshDeviceProfilerResultsCall",
        do_mid_run_dump=True,
        do_cpp_post_process=True,
    )

    with open(PROFILER_LOGS_DIR / "test_get_programs_perf_data_latest.json", "r") as f:
        validate_programs_perf_durations(json.load(f))
    with open(PROFILER_LOGS_DIR / "test_get_programs_perf_data_all.json", "r") as f:
        validate_programs_perf_durations(json.load(f))

    run_gtest_profiler_test(
        test_get_programs_perf_data_binary,
        "GetProgramsPerfDataFixture.TestGetProgramsPerfDataAfterSingleReadMeshDeviceProfilerResultsCall",
        do_mid_run_dump=True,
        do_cpp_post_process=True,
    )

    with open(PROFILER_LOGS_DIR / "test_get_programs_perf_data_latest.json", "r") as f:
        validate_programs_perf_durations(json.load(f))
    with open(PROFILER_LOGS_DIR / "test_get_programs_perf_data_all.json", "r") as f:
        validate_programs_perf_durations(json.load(f))

    run_gtest_profiler_test(
        test_get_programs_perf_data_binary,
        "GetProgramsPerfDataFixture.TestGetProgramsPerfDataAfterMultipleReadMeshDeviceProfilerResultsCalls",
        do_mid_run_dump=True,
        do_cpp_post_process=True,
    )

    with open(PROFILER_LOGS_DIR / "test_get_programs_perf_data_latest.json", "r") as f:
        validate_programs_perf_durations(json.load(f))
    with open(PROFILER_LOGS_DIR / "test_get_programs_perf_data_all.json", "r") as f:
        validate_programs_perf_durations(json.load(f))
