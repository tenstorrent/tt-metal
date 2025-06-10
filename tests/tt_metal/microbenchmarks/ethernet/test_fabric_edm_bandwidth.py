# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import time
import threading

from enum import Enum
from loguru import logger
import pytest
import csv
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tabulate import tabulate
import pandas as pd
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

machine_type_suffix = None

# Global daemon management variables
daemon_process = None
# NOTE: This paths need to be same as the one written in test_fabric_edm.cpp
daemon_pipe_path = "/tmp/tt_metal_fabric_edm_daemon"
daemon_result_pipe_path = "/tmp/tt_metal_fabric_edm_daemon_result"
daemon_lock = threading.Lock()
binary_path = os.environ.get("TT_METAL_HOME", "") + "/build/test/ttnn/unit_tests_ttnn_fabric_edm"

# Global direct execution mode setting (determined once per test session)
_direct_mode_enabled = None


def start_fabric_edm_daemon():
    """Start the fabric EDM daemon if not already running"""
    global daemon_process

    with daemon_lock:
        if daemon_process is not None and daemon_process.poll() is None:
            return  # Daemon already running

        logger.info("Starting fabric EDM daemon...")

        # Clean up any existing pipes
        try:
            os.unlink(daemon_pipe_path)
        except FileNotFoundError:
            pass
        try:
            os.unlink(daemon_result_pipe_path)
        except FileNotFoundError:
            pass

        # Start daemon process
        cmd = [
            binary_path,
            "daemon_mode",
        ]

        env = os.environ.copy()
        env["TT_METAL_ENABLE_ERISC_IRAM"] = "1"
        env["TT_METAL_DEVICE_PROFILER"] = "1"

        daemon_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=None,
            stderr=None,
            text=True,
        )

        # Wait for daemon to create the pipe
        max_wait = 10  # seconds
        wait_time = 0
        while not os.path.exists(daemon_pipe_path) and wait_time < max_wait:
            time.sleep(0.1)
            wait_time += 0.1

        if not os.path.exists(daemon_pipe_path):
            raise RuntimeError("Daemon failed to create communication pipe")

        logger.info(f"Fabric EDM daemon started with PID {daemon_process.pid}")


def stop_fabric_edm_daemon():
    """Stop the fabric EDM daemon"""
    global daemon_process

    with daemon_lock:
        if daemon_process is None:
            return

        logger.info("Stopping fabric EDM daemon...")

        try:
            # Send shutdown command
            with open(daemon_pipe_path, "w") as pipe:
                pipe.write("SHUTDOWN\n")
                pipe.flush()

            # Wait for process to terminate
            daemon_process.wait(timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Force kill if graceful shutdown fails
            if daemon_process.poll() is None:
                daemon_process.kill()
                daemon_process.wait()

        daemon_process = None

        # Clean up pipes
        try:
            os.unlink(daemon_pipe_path)
        except FileNotFoundError:
            pass
        try:
            os.unlink(daemon_result_pipe_path)
        except FileNotFoundError:
            pass

        logger.info("Fabric EDM daemon stopped")


def send_test_to_daemon(test_mode, test_params_str):
    """Send test parameters to daemon and get result"""

    # Create result pipe if it doesn't exist
    try:
        os.mkfifo(daemon_result_pipe_path, 0o666)
    except FileExistsError:
        pass

    # Send test command
    command = f"TEST:{test_mode}:{test_params_str}\n"

    with open(daemon_pipe_path, "w") as pipe:
        pipe.write(command)
        pipe.flush()

    # Read result
    with open(daemon_result_pipe_path, "r") as result_pipe:
        result_line = result_pipe.readline().strip()
        return int(result_line)


def get_direct_mode():
    return _direct_mode_enabled


def set_direct_mode(enabled):
    global _direct_mode_enabled
    _direct_mode_enabled = enabled


def update_machine_type_suffix(machine_type: str):
    global machine_type_suffix
    machine_type_suffix = machine_type


def reset_machine_type_suffix():
    global machine_type_suffix
    machine_type_suffix = None


# Python enum mirroring test_fabric_edm_common.hpp
class FabricTestMode(Enum):
    Linear = 0
    HalfRing = 1
    FullRing = 2
    SaturateChipToChipRing = 3
    RingAsLinear = 4


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def summarize_to_csv(
    test_name,
    packet_size,
    line_size,
    num_links,
    disable_sends_for_interior_workers,
    unidirectional,
    bandwidth,
    packets_per_second,
    *,
    noc_message_type,
    senders_are_unidirectional,
):
    """Write test results to a CSV file organized by packet size"""
    csv_path = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/bandwidth_summary.csv")

    # Create header if file doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Test Name",
                    "Noc Message Type",
                    "Packet Size",
                    "Line Size",
                    "Num Links",
                    "Disable Interior Workers",
                    "Unidirectional",
                    "Senders Are Unidirectional",
                    "Bandwidth (B/c)",
                    "Packets/Second",
                ]
            )

    # Append results
    with open(csv_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                test_name,
                noc_message_type,
                packet_size,
                line_size,
                num_links,
                disable_sends_for_interior_workers,
                unidirectional,
                senders_are_unidirectional,
                bandwidth,
                packets_per_second,
            ]
        )


def read_golden_results(
    test_name,
    packet_size,
    line_size,
    num_links,
    disable_sends_for_interior_workers,
    unidirectional,  # traffic at fabric level
    *,
    noc_message_type,
    senders_are_unidirectional=False,  # coming out of any given worker
):
    """Print a summary table of all test results by packet size"""
    csv_path = os.path.join(
        os.environ["TT_METAL_HOME"],
        f"tests/tt_metal/microbenchmarks/ethernet/fabric_edm_bandwidth_golden{'_' + machine_type_suffix if machine_type_suffix is not None else ''}.csv",
    )

    if not os.path.exists(csv_path):
        logger.warning("No golden data found")
        return 0, 0

    df = pd.read_csv(csv_path)
    df = df.replace({float("nan"): None})
    results = df[
        (df["Test Name"] == test_name)
        & (df["Noc Message Type"] == noc_message_type)
        & (df["Packet Size"] == packet_size)
        & (df["Line Size"] == str(line_size))
        & (df["Num Links"] == str(num_links))
        & (df["Disable Interior Workers"] == str(disable_sends_for_interior_workers))
        & (df["Unidirectional"] == str(unidirectional))
        & (df["Senders Are Unidirectional"] == str(senders_are_unidirectional))
    ]

    if len(results["Bandwidth (B/c)"]) == 0 or len(results["Packets/Second"]) == 0:
        logger.error(
            f"No golden data found for tests_name={test_name} noc_message_type={noc_message_type} packet_size={packet_size} line_size={line_size} num_links={num_links} disable_sends_for_interior_workers={disable_sends_for_interior_workers} unidirectional={unidirectional}"
        )
        assert (
            len(results["Bandwidth (B/c)"]) == 0 and len(results["Packets/Second"]) == 0
        ), "Golden data may be incorrect or corrupted. One of `Bandwidth (B/c)` or `Packets/Second` was missing but not both. Either both should be present or both should be missing."
        return 0, 0

    bandwidth = results["Bandwidth (B/c)"].values[0]
    pps = results["Packets/Second"].values[0]

    return float(bandwidth) if bandwidth is not None else None, float(pps) if pps is not None else None


def profile_results(
    zone_name,
    packets_per_src_chip,
    line_size,
    packet_size,
    fabric_mode,
    disable_sends_for_interior_workers,
    unidirectional=False,
):
    freq_hz = get_device_freq() * 1000.0 * 1000.0
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    setup.timerAnalysis = {
        zone_name: {
            "across": "core",
            "type": "adjacent",
            # "type": "session_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zone_name": zone_name},
            "end": {"core": "ANY", "risc": "ANY", "zone_name": zone_name},
        },
    }
    devices_data = import_log_run_stats(setup)
    devices = list(devices_data["devices"].keys())

    # MAIN-TEST-BODY
    main_loop_cycles = []
    for device in devices:
        main_loop_cycle = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][zone_name]["stats"]["Max"]
        main_loop_cycles.append(main_loop_cycle)

    def get(x):
        if type(x) is list:
            assert len(x) == 2, "Line size must be a list of two elements"
            assert x[0] == x[1], "Line size must be the same for both axes"
            return x[0]
        return x

    max_line_size = line_size
    if type(line_size) is list:
        max_line_size = int(max(line_size))
    if fabric_mode == FabricTestMode.FullRing:
        traffic_streams_through_boundary = max_line_size - 1
    elif fabric_mode == FabricTestMode.SaturateChipToChipRing:
        traffic_streams_through_boundary = 3
    else:
        traffic_streams_through_boundary = max_line_size // 2
        if get(disable_sends_for_interior_workers):
            traffic_streams_through_boundary = 1
        if get(unidirectional):
            traffic_streams_through_boundary = 1
    total_packets_sent = packets_per_src_chip * traffic_streams_through_boundary
    total_byte_sent = total_packets_sent * packet_size
    bandwidth = total_byte_sent / max(main_loop_cycles)
    packets_per_second = total_packets_sent / max(main_loop_cycles) * freq_hz
    bytes_per_GB = 1000000000
    bandwidth_GB_s = (bandwidth * freq_hz) / bytes_per_GB
    logger.info("main_loop_cycles: {} ", max(main_loop_cycles))
    logger.info("bandwidth: {} GB/s", bandwidth_GB_s)

    return bandwidth, packets_per_second


def process_results(
    *,
    test_name,
    zone_name_inner,
    zone_name_main,
    is_unicast,
    num_messages,
    noc_message_type,
    num_links,
    line_size,
    packet_size,
    fabric_mode,
    disable_sends_for_interior_workers,
    unidirectional=False,
    senders_are_unidirectional=False,
):
    bandwidth_inner_loop, packets_per_second_inner_loop = profile_results(
        zone_name_inner,
        num_messages,
        line_size,
        packet_size,
        fabric_mode,
        disable_sends_for_interior_workers,
        unidirectional=unidirectional,
    )
    bandwidth, packets_per_second = profile_results(
        zone_name_main,
        num_messages,
        line_size,
        packet_size,
        fabric_mode,
        disable_sends_for_interior_workers,
        unidirectional=unidirectional,
    )
    logger.info("bandwidth_inner_loop: {} B/c", bandwidth_inner_loop)
    logger.info("bandwidth: {} B/c", bandwidth)
    logger.info("packets_per_second_inner_loop: {} pps", packets_per_second_inner_loop)
    logger.info("packets_per_second: {} pps", packets_per_second)

    # Add summary to CSV
    summarize_to_csv(
        test_name,
        packet_size,
        line_size,
        num_links,
        disable_sends_for_interior_workers,
        unidirectional,
        bandwidth,
        packets_per_second,
        noc_message_type=noc_message_type,
        senders_are_unidirectional=senders_are_unidirectional,
    )
    expected_bw, expected_pps = read_golden_results(
        test_name,
        packet_size,
        line_size,
        num_links,
        disable_sends_for_interior_workers,
        unidirectional,
        noc_message_type=noc_message_type,
        senders_are_unidirectional=senders_are_unidirectional,
    )
    bw_threshold_general = 0.07
    pps_threshold_general = 0.01
    if packet_size <= 2048 and fabric_mode != FabricTestMode.Linear:
        bw_threshold_general = 0.12
    ## These seem to be a little more noisy so for now we widen the threshold to have test stability
    bw_threshold_fused_write_atomic = 0.15
    pps_threshold_fused_write_atomic = 0.03
    use_general_threshold = (
        noc_message_type != "noc_fused_unicast_write_flush_atomic_inc"
        and noc_message_type != "noc_fused_unicast_write_no_flush_atomic_inc"
    )

    bw_threshold = bw_threshold_general if use_general_threshold else bw_threshold_fused_write_atomic
    pps_threshold = pps_threshold_general if use_general_threshold else pps_threshold_fused_write_atomic

    mega_packets_per_second = packets_per_second / 1000000
    expected_Mpps = expected_pps / 1000000 if expected_pps is not None else None
    assert (
        expected_bw - bw_threshold <= bandwidth <= expected_bw + bw_threshold
    ), f"Bandwidth mismatch. expected: {expected_bw} B/c, actual: {bandwidth} B/c"
    if expected_Mpps is not None:
        assert (
            expected_Mpps - expected_pps <= mega_packets_per_second <= expected_Mpps + expected_pps
        ), f"Packets per second mismatch. expected: {expected_Mpps} Mpps, actual: {mega_packets_per_second} Mpps"


def build_test_params_str(
    separator,
    *,
    is_unicast,
    num_messages,
    noc_message_type,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_mode,
    disable_sends_for_interior_workers,
    unidirectional=False,
    senders_are_unidirectional=False,
    test_mode="1_fabric_instance",
    num_cluster_rows=0,
    num_cluster_cols=0,
):
    test_params_str = test_mode

    def validate_two_entries(x):
        assert type(x) is list and len(x) == 2

    def validate_non_zero(x):
        assert x != 0, f"Expected non-zero value, got {x}"

    def normalize(x):
        if type(x) is not list:
            x = [x]
        return x

    def append_tokens(x):
        nonlocal test_params_str
        if type(x) is str:
            test_params_str += f"{separator}{x}"
        else:
            test_params_str += f"{separator}{int(x)}"

    def apply(x, func):
        x = normalize(x)
        for i in x:
            func(i)

    both_axes_active = test_mode == "1D_fabric_on_mesh_multi_axis"
    for x in [both_axes_active, is_unicast, noc_message_type, num_messages]:
        apply(x, append_tokens)

    for x in [num_links, line_size, num_op_invocations, packet_size, num_messages]:
        apply(x, validate_non_zero)

    if both_axes_active:
        for x in [num_links, line_size, disable_sends_for_interior_workers, unidirectional, senders_are_unidirectional]:
            try:
                validate_two_entries(x)
            except Exception as e:
                raise ValueError(f"Expected list of length 2, got {x} with type {type(x)}. {e}")

    arg_order = [
        num_links,
        num_op_invocations,
        line_sync,
        line_size,
        packet_size,
        fabric_mode,
        disable_sends_for_interior_workers,
        unidirectional,
        senders_are_unidirectional,
    ]
    for x in arg_order:
        apply(x, append_tokens)

    if test_mode == "1D_fabric_on_mesh" or both_axes_active:
        append_tokens(num_cluster_rows)
        append_tokens(num_cluster_cols)
        append_tokens(0)

    test_params_str += f"{separator}"

    return test_params_str


def run_fabric_edm(
    *,
    is_unicast,
    num_messages,
    noc_message_type,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_mode,
    disable_sends_for_interior_workers,
    unidirectional=False,
    senders_are_unidirectional=False,
    test_mode="1_fabric_instance",
    num_cluster_rows=0,
    num_cluster_cols=0,
):
    if test_mode == "1_fabric_instance":
        assert num_cluster_rows == 0 and num_cluster_cols == 0
        test_name = f"{'unicast' if is_unicast else 'mcast'}_{fabric_mode.name}"
    elif test_mode == "1D_fabric_on_mesh":
        test_name = (
            f"{'unicast' if is_unicast else 'mcast'}_{fabric_mode.name}_{num_cluster_rows}x{num_cluster_cols}_mesh"
        )
    elif test_mode == "1D_fabric_on_mesh_multi_axis":
        test_name = f"{'unicast' if is_unicast else 'mcast'}_{fabric_mode.name}_all_rows_all_cols_mesh"

    else:
        raise ValueError(f"Invalid test mode: {test_mode}")

    logger.warning("removing file profile_log_device.csv")
    subprocess.run(["rm", "-rf", f"{os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv"])

    enable_persistent_kernel_cache()

    use_direct_exec = get_direct_mode()

    # Create the args string for daemon
    test_params_str = build_test_params_str(
        " " if use_direct_exec else "|",
        is_unicast=is_unicast,
        num_messages=num_messages,
        noc_message_type=noc_message_type,
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_mode.value,
        disable_sends_for_interior_workers=disable_sends_for_interior_workers,
        unidirectional=unidirectional,
        senders_are_unidirectional=senders_are_unidirectional,
        test_mode=test_mode,
        num_cluster_rows=num_cluster_rows,
        num_cluster_cols=num_cluster_cols,
    )

    if not use_direct_exec:
        try:
            # Start daemon if not already running
            start_fabric_edm_daemon()

            logger.info(f"Sending test to daemon: {test_mode}:{test_params_str}")
            rc = send_test_to_daemon(test_mode, test_params_str)

        except Exception as e:
            logger.warning(f"Daemon mode failed: {e}, falling back to direct execution")
    else:
        # Fallback to original direct execution
        cmd = f"TT_METAL_ENABLE_ERISC_IRAM=1 TT_METAL_DEVICE_PROFILER=1 \
                    {binary_path} \
                    {test_params_str}"
        logger.info(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=False)
        rc = result.returncode

    disable_persistent_kernel_cache()
    if rc != 0:
        # Handle exit codes differently for daemon vs direct execution
        reset_machine_type_suffix()
        if rc == 1:
            pytest.skip("Skipping test because it only works with T3000")
            return
        logger.info(f"Error in running the test {rc}")
        assert False

    zone_name_inner = "MAIN-TEST-BODY"
    zone_name_main = "MAIN-TEST-BODY"

    process_results(
        test_name=test_name,
        zone_name_inner=zone_name_inner,
        zone_name_main=zone_name_main,
        is_unicast=is_unicast,
        num_messages=num_messages,
        noc_message_type=noc_message_type,
        num_links=num_links,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_mode,
        disable_sends_for_interior_workers=disable_sends_for_interior_workers,
        unidirectional=unidirectional,
        senders_are_unidirectional=senders_are_unidirectional,
    )

    # Reset for the next test case
    reset_machine_type_suffix()


@pytest.fixture(scope="session", autouse=True)
def initialize_daemon_mode(request):
    """Initialize global daemon mode setting once per test session"""
    # Check pytest command line option first, then fall back to environment variable
    direct_exec_mode_enabled = hasattr(request.config.option, "direct_exec") and request.config.option.direct_exec

    # Set global daemon mode
    set_direct_mode(direct_exec_mode_enabled)

    if direct_exec_mode_enabled:
        logger.info("Fabric EDM daemon mode disabled, using direct execution")
        yield
    else:
        logger.info("Fabric EDM daemon mode enabled for this test session")
        start_fabric_edm_daemon()
        yield
        logger.info("Stopping fabric EDM daemon after test session")
        stop_fabric_edm_daemon()


@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("line_size, num_links", [(4, 1), (4, 2), (4, 3), (4, 4)])
def test_fabric_edm_mcast_half_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.HalfRing,
        disable_sends_for_interior_workers=False,
        unidirectional=False,
        senders_are_unidirectional=False,
    )


@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_4chip_one_link_mcast_full_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.FullRing,
        disable_sends_for_interior_workers=False,
    )


@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [2, 3, 4])
@pytest.mark.parametrize("packet_size", [4096])
def test_fabric_4chip_multi_link_mcast_full_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.FullRing,
        disable_sends_for_interior_workers=False,
    )


@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [8])
@pytest.mark.parametrize("num_links", [1, 2, 3, 4])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_8chip_multi_link_edm_mcast_half_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.HalfRing,
        disable_sends_for_interior_workers=False,
    )


@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [8])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_8chip_one_link_edm_mcast_full_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.FullRing,
        disable_sends_for_interior_workers=False,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [8])
@pytest.mark.parametrize("num_links", [2, 3, 4])
@pytest.mark.parametrize("packet_size", [4096])
def test_fabric_8chip_multi_link_edm_mcast_full_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.FullRing,
        disable_sends_for_interior_workers=False,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1, 2, 3, 4])
@pytest.mark.parametrize("packet_size", [4096])
def test_fabric_4chip_multi_link_edm_unicast_full_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.FullRing,
        disable_sends_for_interior_workers=False,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_4_chip_one_link_mcast_saturate_chip_to_chip_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.SaturateChipToChipRing,
        disable_sends_for_interior_workers=False,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("num_links", [2, 3, 4])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("packet_size", [4096])
def test_fabric_4_chip_multi_link_mcast_saturate_chip_to_chip_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.SaturateChipToChipRing,
        disable_sends_for_interior_workers=False,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [2])
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear])
@pytest.mark.parametrize("num_cluster_cols", [4])
def test_fabric_t3k_4chip_cols_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
    num_cluster_cols,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        unidirectional=False,
        senders_are_unidirectional=True,
        test_mode="1D_fabric_on_mesh",
        num_cluster_rows=0,
        num_cluster_cols=num_cluster_cols,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear])
@pytest.mark.parametrize("num_cluster_rows", [2])
def test_fabric_t3k_4chip_rows_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
    num_cluster_rows,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        unidirectional=False,
        senders_are_unidirectional=True,
        test_mode="1D_fabric_on_mesh",
        num_cluster_rows=num_cluster_rows,
        num_cluster_cols=0,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [[4, 2]])  # first entry is row size (X dim), second entry is col size (Y dim)
@pytest.mark.parametrize("num_links", [[1, 1]])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear])
@pytest.mark.parametrize("num_cluster_rows,num_cluster_cols", [(2, 4)])
def test_fabric_t3k_all_rows_and_cols_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
    num_cluster_rows,
    num_cluster_cols,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=[False, False],
        unidirectional=[False, False],
        senders_are_unidirectional=[True, True],
        test_mode="1D_fabric_on_mesh_multi_axis",
        num_cluster_rows=num_cluster_rows,
        num_cluster_cols=num_cluster_cols,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.sanity_6u
@pytest.mark.parametrize("is_unicast", [False])
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [8])
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("packet_size", [2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.FullRing, FabricTestMode.Linear])
@pytest.mark.parametrize("num_cluster_cols", [4])
def test_fabric_6u_4chip_cols_mcast_bw(
    is_unicast,
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
    num_cluster_cols,
):
    is_ring = fabric_test_mode == FabricTestMode.FullRing
    if is_ring:
        pytest.skip("Baseline numbers not yet available for full-6u ring fabric test mode")
    update_machine_type_suffix("6u")
    run_fabric_edm(
        is_unicast=is_unicast,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        senders_are_unidirectional=not is_ring,
        test_mode="1D_fabric_on_mesh",
        num_cluster_rows=0,
        num_cluster_cols=num_cluster_cols,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.sanity_6u
@pytest.mark.parametrize("is_unicast", [False])
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("packet_size", [2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.FullRing, FabricTestMode.Linear])
@pytest.mark.parametrize("num_cluster_rows", [8])
def test_fabric_6u_4chip_rows_mcast_bw(
    is_unicast,
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
    num_cluster_rows,
):
    is_ring = fabric_test_mode == FabricTestMode.FullRing
    if is_ring:
        pytest.skip("Baseline numbers not yet available for full-6u ring fabric test mode")
    update_machine_type_suffix("6u")
    run_fabric_edm(
        is_unicast=is_unicast,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        senders_are_unidirectional=not is_ring,
        test_mode="1D_fabric_on_mesh",
        num_cluster_rows=num_cluster_rows,
        num_cluster_cols=0,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("is_unicast", [True, False])
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [[4, 8]])  # first entry is row size (X dim), second entry is col size (Y dim)
@pytest.mark.parametrize("num_links", [[4, 4], [1, 1]])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.FullRing, FabricTestMode.Linear])
@pytest.mark.parametrize("num_cluster_rows,num_cluster_cols", [(8, 4)])
def test_fabric_6u_all_rows_and_cols_mcast_bw(
    is_unicast,
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
    num_cluster_rows,
    num_cluster_cols,
):
    is_ring = fabric_test_mode == FabricTestMode.FullRing
    update_machine_type_suffix("6u")
    run_fabric_edm(
        is_unicast=is_unicast,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=[False, False],
        unidirectional=[False, False],
        senders_are_unidirectional=[not is_ring, not is_ring],
        test_mode="1D_fabric_on_mesh_multi_axis",
        num_cluster_rows=num_cluster_rows,
        num_cluster_cols=num_cluster_cols,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_4chip_one_link_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        unidirectional=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_4chip_one_link_bidirectional_single_producer_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=True,
        unidirectional=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_4chip_one_link_unidirectional_single_producer_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=True,
        unidirectional=True,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [2, 3, 4])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_4chip_two_link_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [2])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_non_forwarding_unicast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [2])
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_two_link_non_forwarding_unicast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_forwarding_unicast_multiproducer_multihop_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.ubench_quick_tests
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_forwarding_unicast_single_producer_multihop_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=True,
        unidirectional=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_forwarding_unicast_unidirectional_single_producer_multihop_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=True,
        unidirectional=True,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("noc_message_type", ["noc_unicast_flush_atomic_inc", "noc_unicast_no_flush_atomic_inc"])
@pytest.mark.parametrize("packet_size", [16])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_forwarding_unicast_single_producer_multihop_atomic_inc_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    noc_message_type,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type=noc_message_type,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=True,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("is_unicast", [False, True])
@pytest.mark.parametrize("disable_sends_for_interior_workers", [False, True])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("unidirectional", [False, True])
@pytest.mark.parametrize(
    "noc_message_type", ["noc_fused_unicast_write_flush_atomic_inc", "noc_fused_unicast_write_no_flush_atomic_inc"]
)
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_multihop_fused_write_atomic_inc_bw(
    is_unicast,
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    disable_sends_for_interior_workers,
    noc_message_type,
    unidirectional,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=is_unicast,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type=noc_message_type,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=disable_sends_for_interior_workers,
        unidirectional=unidirectional,
        senders_are_unidirectional=True,
    )


def print_bandwidth_summary():
    """Print a summary table of all test results by packet size"""
    csv_path = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/bandwidth_summary.csv")

    if not os.path.exists(csv_path):
        logger.warning("No bandwidth summary data found")
        return

    df = pd.read_csv(csv_path)

    # Sort by test name and packet size
    df = df.sort_values(
        [
            "Test Name",
            "Noc Message Type",
            "Packet Size",
            "Line Size",
            "Num Links",
            "Disable Interior Workers",
            "Unidirectional",
            "Senders Are Unidirectional",
        ]
    )

    # Format table with raw values
    table = tabulate(
        df,
        headers=[
            "Test Name",
            "Noc Message Type",
            "Packet Size",
            "Line Size",
            "Num Links",
            "Disable Interior Workers",
            "Unidirectional",
            "Senders Are Unidirectional",
            "Bandwidth (B/c)",
            "Packets/Second",
        ],
        tablefmt="grid",
        floatfmt=".2f",
    )
    logger.info("\nBandwidth Test Results:\n{}", table)


@pytest.fixture(scope="session", autouse=True)
def print_summary_at_end(request):
    """Print bandwidth summary at end of session"""
    # Delete old CSV file at start
    csv_path = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/bandwidth_summary.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
        logger.info("Removed old bandwidth summary file")

    yield
    print_bandwidth_summary()
