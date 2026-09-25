# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# You need to run these tests from python_env that is created with ./create_venv.sh
# You also need to install everything needed to run tt-triage.py in that environment
# Run manually ./tools/tt-triage.py --help to see if it works and install requirements

from collections import deque
from dataclasses import fields
from datetime import timedelta
import os
import signal
import subprocess
import sys
import threading
import time

import pytest


metal_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
triage_script = os.path.join(metal_home, "tools", "tt-triage.py")
triage_home = os.path.join(metal_home, "tools", "triage")


# Add triage tools directory to Python path
sys.path.insert(0, triage_home)


import triage
from triage import CheckType, run_script, ScriptArguments
from triage_hw_utils import device_has_firmware
from ttexalens.context import Context
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.coordinate import OnChipCoordinate


triage.progress_disabled = True  # Disable progress bars for tests


def logged_errors() -> list[str]:
    """The failed checks reported so far, formatted the way triage prints them."""
    return [check.formatted_message for check in triage.CHECKS if check.type is CheckType.ERROR]


# Mapping of hang application paths to their expected test results
HANG_APP_ADD_2_INTEGERS = "tools/tests/triage/hang_apps/add_2_integers_hang/triage_hang_app_add_2_integers_hang"
HANG_APP_TTNN_ADD_INTEGERS = (
    "tools/tests/triage/hang_apps/ttnn_add_integers_hang/triage_hang_app_ttnn_add_integers_hang"
)
HANG_APP_MESH_SOCKET = "tools/tests/triage/hang_apps/mesh_socket_hang/mesh_socket_hang.py"

MESH_SOCKET_FIFO_SIZE = 8192

SIMULATOR_HANG_DEADLINE_SECONDS = int(os.environ.get("TT_TRIAGE_SIMULATOR_HANG_DEADLINE_SECONDS", "900"))
SIMULATOR_POLL_INTERVAL_SECONDS = 2
APP_SHUTDOWN_TIMEOUT_SECONDS = 20

HANG_APP_EXPECTED_RESULTS = {
    HANG_APP_ADD_2_INTEGERS: {
        "lightweight_asserts": {
            "kernel_name": "add_2_tiles_hang",
            "compute_cores_hang": True,
            "first_callstack_file": "add_2_tiles_hang.cpp",
            "first_callstack_line": 40,
        },
        "callstacks": {
            "device_to_check": 0,
            "location_to_check": "0,0",  # Only check this core location
            "cores_to_check": {
                "trisc0": {
                    "file": "add_2_tiles_hang.cpp",
                    "line": 40,
                },
                "trisc1": {
                    "file": "add_2_tiles_hang.cpp",
                    "line": 40,
                },
                "trisc2": {
                    "file": "add_2_tiles_hang.cpp",
                    "line": 40,
                },
            },
        },
    },
    HANG_APP_TTNN_ADD_INTEGERS: {
        "lightweight_asserts": {
            "kernel_name": "add_2_tiles_hang",
            "compute_cores_hang": True,
            "first_callstack_file": "add_2_tiles_hang.cpp",
            "first_callstack_line": 40,
        },
        "callstacks": {
            "device_to_check": 0,
            "location_to_check": "0,0",
            "cores_to_check": {
                "trisc0": {
                    "file": "add_2_tiles_hang.cpp",
                    "line": 40,
                },
                "trisc1": {
                    "file": "add_2_tiles_hang.cpp",
                    "line": 40,
                },
                "trisc2": {
                    "file": "add_2_tiles_hang.cpp",
                    "line": 40,
                },
            },
        },
        "running_operations": {
            "expected_op_name_contains": "AddIntegersHang",
            "assert_no_na": True,
        },
    },
}


class AppOutput:
    RETAINED_LINES = 20000

    def __init__(self, proc: subprocess.Popen):
        self._lock = threading.Lock()
        self._chunks: deque[str] = deque(maxlen=self.RETAINED_LINES)
        for stream in (proc.stdout, proc.stderr):
            threading.Thread(target=self._drain, args=(stream,), daemon=True).start()

    def _drain(self, stream) -> None:
        try:
            for line in iter(stream.readline, b""):
                with self._lock:
                    self._chunks.append(line.decode("utf-8", errors="replace"))
        except (ValueError, OSError):
            pass  # the pipe was closed under us while the app was being torn down

    def text(self) -> str:
        with self._lock:
            return "".join(self._chunks)

    def print(self) -> None:
        # Pytest will only display this if the test fails.
        print("\n=== Application output ===")
        print(self.text() or "(empty)")


def compute_risc_names(location: OnChipCoordinate, neo_id: int | None) -> set[str]:
    return {
        risc.risc_location.risc_name
        for risc in location.device.get_block(location).all_riscs
        if risc.risc_location.neo_id == neo_id and risc.risc_location.risc_name.startswith("trisc")
    }


def hang_is_in_place(context: Context, expected_results: dict) -> bool:
    expected = expected_results.get("callstacks")
    if not expected:
        return True

    location_str = expected.get("location_to_check")
    wanted = len(expected.get("cores_to_check", {})) or 1
    for device in context.devices.values():
        location = OnChipCoordinate.create(location_str, device)
        halted = 0
        for risc_debug in location.device.get_block(location).all_riscs:
            try:
                if risc_debug.is_halted():
                    halted += 1
            except Exception:
                continue
        if halted >= wanted:
            return True
    return False


def wait_for_simulated_hang(proc: subprocess.Popen, output: AppOutput, app: str, expected_results: dict) -> Context:
    deadline = time.monotonic() + SIMULATOR_HANG_DEADLINE_SECONDS
    context: Context | None = None
    consecutive_ready = 0
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"{app} exited with {proc.returncode} before its device hung.\n{output.text()}")
        try:
            if context is None:
                context = init_ttexalens()
            consecutive_ready = consecutive_ready + 1 if hang_is_in_place(context, expected_results) else 0
            if consecutive_ready >= 2:
                return context
        except Exception:
            consecutive_ready = 0

        if time.monotonic() > deadline:
            stage = "start its simulator" if context is None else "reach its hang"
            raise RuntimeError(
                f"{app} did not {stage} within {SIMULATOR_HANG_DEADLINE_SECONDS}s. Raise "
                f"TT_TRIAGE_SIMULATOR_HANG_DEADLINE_SECONDS if the simulator is simply slow.\n{output.text()}"
            )
        time.sleep(SIMULATOR_POLL_INTERVAL_SECONDS)


@pytest.fixture(scope="class")
def cause_hang_with_app(request):
    global metal_home

    app, args, app_configuration, timeout = request.param
    os.environ.pop("TT_METAL_LOGS_PATH", None)
    min_devices = app_configuration.get("min_devices", 1)

    on_simulator = bool(os.environ.get("TT_METAL_SIMULATOR"))
    if on_simulator and app_configuration.get("auto_timeout", False):
        pytest.skip("Automatic hang detection cannot be exercised on a simulator")

    if on_simulator and min_devices > 1:
        pytest.skip(f"{app} needs {min_devices} chips, and a simulator hosts one config")

    if not on_simulator:
        request.cls.exalens_context = init_ttexalens()
        if len(request.cls.exalens_context.devices) < min_devices:
            pytest.skip(f"{app} needs {min_devices} chips")

    if app.endswith(".py"):
        # Python apps live in the source tree and need this venv's interpreter, not the system one.
        cmd = [sys.executable, os.path.join(metal_home, app)]
    else:
        cmd = [os.path.join(metal_home, "build", app)]
    proc = subprocess.Popen(
        cmd + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **app_configuration.get("env", {})},
    )

    output = AppOutput(proc)
    try:
        expected_results = app_configuration.get("expected_results", {})
        if on_simulator:
            request.cls.exalens_context = wait_for_simulated_hang(proc, output, app, expected_results)
        else:
            auto_timeout = app_configuration.get("auto_timeout", False)
            # auto_timeout apps exit 0 once they detect their own hang; expect_running ones stay wedged.
            expect_running = app_configuration.get("expect_running", False)
            if auto_timeout or expect_running:
                # Wait for the application to hang itself
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass

                # Check if the process has exited
                if proc.returncode != (None if expect_running else 0):
                    # Print process output for debugging
                    print("The application did not hang as expected.")
                    output.print()
                    raise RuntimeError("The application did not hang as expected.")
            else:
                time.sleep(timeout)

        request.cls.app_configuration = app_configuration
        request.cls.expected_results = expected_results
        if app_configuration.get("env", {}).get("TT_METAL_LOGS_PATH"):
            metal_logs_path = app_configuration["env"]["TT_METAL_LOGS_PATH"]
            os.environ["TT_METAL_LOGS_PATH"] = metal_logs_path

        yield
    finally:
        # Clean up the hung application
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=APP_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        output.print()

        # Reset the device state after the hang if set in environment
        if os.environ.get("TT_METAL_RESET_DEVICE_AFTER_HANG", "0") == "1":
            subprocess.run(["tt-smi", "-r"], check=True)


@pytest.mark.parametrize(
    "cause_hang_with_app",
    [
        (
            # Manual hang detection with timeout from outside
            HANG_APP_ADD_2_INTEGERS,
            [],
            {
                "expected_results": HANG_APP_EXPECTED_RESULTS[HANG_APP_ADD_2_INTEGERS],
            },
            10,
        ),
        (
            # Automatic hang detection with timeout inside the app and serialization of Inspector RPC data, fast dispatch
            HANG_APP_ADD_2_INTEGERS,
            [],
            {
                "auto_timeout": True,
                "env": {
                    "TT_METAL_OPERATION_TIMEOUT_SECONDS": "0.5",
                    "TT_METAL_LOGS_PATH": "/tmp/tt-metal/triage-test",
                },
                "expected_results": HANG_APP_EXPECTED_RESULTS[HANG_APP_ADD_2_INTEGERS],
            },
            60,
        ),
        (
            # Automatic hang detection with timeout inside the app and serialization of Inspector RPC data, slow dispatch
            HANG_APP_ADD_2_INTEGERS,
            [],
            {
                "auto_timeout": True,
                "env": {
                    "TT_METAL_OPERATION_TIMEOUT_SECONDS": "0.5",
                    "TT_METAL_LOGS_PATH": "/tmp/tt-metal/inspector",
                    "TT_METAL_SLOW_DISPATCH_MODE": "1",
                },
                "expected_results": HANG_APP_EXPECTED_RESULTS[HANG_APP_ADD_2_INTEGERS],
            },
            60,
        ),
        (
            # TTNN-dispatched hang: auto detection, fast dispatch
            HANG_APP_TTNN_ADD_INTEGERS,
            [],
            {
                "auto_timeout": True,
                "env": {
                    "TT_METAL_OPERATION_TIMEOUT_SECONDS": "0.5",
                    "TT_METAL_LOGS_PATH": "/tmp/tt-metal/triage-test-ttnn",
                },
                "expected_results": HANG_APP_EXPECTED_RESULTS[HANG_APP_TTNN_ADD_INTEGERS],
            },
            60,
        ),
        (
            # TTNN-dispatched hang: auto detection, slow dispatch
            HANG_APP_TTNN_ADD_INTEGERS,
            [],
            {
                "auto_timeout": True,
                "env": {
                    "TT_METAL_OPERATION_TIMEOUT_SECONDS": "0.5",
                    "TT_METAL_LOGS_PATH": "/tmp/tt-metal/inspector-ttnn",
                    "TT_METAL_SLOW_DISPATCH_MODE": "1",
                },
                "expected_results": HANG_APP_EXPECTED_RESULTS[HANG_APP_TTNN_ADD_INTEGERS],
            },
            60,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("cause_hang_with_app")
class TestTriage:
    app_configuration: dict
    exalens_context: Context

    def test_triage_help(self):
        global triage_script

        result = subprocess.run(
            [triage_script, "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert len(result.stderr) == 0
        stdout = result.stdout.decode("utf-8")
        assert "Usage:" in stdout
        assert "triage " in stdout
        assert "Options:" in stdout

    def test_triage_executes_no_errors(self):
        global triage_script

        result = subprocess.run(
            [triage_script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert len(result.stderr) == 0

    def test_triage_verbosity(self):
        global triage_script

        result = subprocess.run(
            [triage_script, "--verbosity=4", "--run=test_output"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert len(result.stderr) == 0

    def test_triage_with_noc_0(self):
        global triage_script

        result = subprocess.run(
            [triage_script, "--noc-id=0", "--run=test_output"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert len(result.stderr) == 0

    def test_triage_skip_version_check(self):
        global triage_script

        result = subprocess.run(
            [triage_script, "--skip-version-check", "--run=test_output"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert len(result.stderr) == 0

    def test_triage_print_script_times(self):
        global triage_script

        result = subprocess.run(
            [triage_script, "--print-script-times", "--run=test_output"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert len(result.stderr) == 0

    def test_triage_verbose(self):
        global triage_script

        result = subprocess.run(
            [triage_script, "-vvv", "--run=test_output"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert len(result.stderr) == 0

    def test_triage_disable_colors(self):
        global triage_script

        result = subprocess.run(
            [triage_script, "--disable-colors", "--run=test_output"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert len(result.stderr) == 0

    # Tests below test individual triage scripts

    def test_check_arc(self):
        result = self.run_triage_script("check_arc.py")

        if not self.any_device_has_firmware():
            assert result is None, "check_arc.py has no ARC to report on, so it should return nothing"
            pytest.skip("No device exposes firmware that triage can read")
        assert result is not None, "Expected non-None result from check_arc.py"
        for check in result:
            assert check.result is not None, "Expected non-None result for each ARC check"

            assert (
                check.result.location == check.device_description.device.arc_block.location
            ), f"Incorrect ARC location: {check.result.location}"
            assert 0 < check.result.clock_mhz < 10000, f"Invalid ARC clock: {check.result.clock_mhz}"
            assert (
                timedelta(seconds=0) < check.result.uptime < timedelta(days=8 * 365)
            ), f"Invalid ARC uptime: {check.result.uptime}"

    def test_device_telemetry(self):
        result = self.run_triage_script("device_telemetry.py")

        if not self.any_device_has_firmware():
            assert result is None, "device_telemetry.py has no telemetry to report, so it should return nothing"
            pytest.skip("No device exposes firmware that triage can read")
        self.assert_no_errors_or_none_in_result(result)

    def test_firmware_versions(self):
        result = self.run_triage_script("firmware_versions.py")

        if not self.any_device_has_firmware():
            assert result is None, "firmware_versions.py has no firmware to report, so it should return nothing"
            pytest.skip("No device exposes firmware that triage can read")
        self.assert_no_errors_or_none_in_result(result)

    def test_check_binary_integrity(self):
        self.run_triage_script("check_binary_integrity.py")

    def test_check_cb_inactive(self):
        self.run_triage_script("check_cb_inactive.py")

    def test_check_core_magic(self):
        self.run_triage_script("check_core_magic.py")

    def test_check_l1_status(self):
        self.run_triage_script("check_l1_status.py")

    def test_check_eth_status(self):
        self.run_triage_script("check_eth_status.py")

    def test_check_noc_locations(self):
        self.run_triage_script("check_noc_locations.py")

    def test_check_noc_status(self):
        self.run_triage_script("check_noc_status.py", assert_failure_checks=False)

        # Some mismatches may occur on unused cores.
        non_state_failures = [failure for failure in logged_errors() if "Mismatched state" not in failure]
        assert (
            len(non_state_failures) == 0
        ), f"Check NOC status check failed with {len(non_state_failures)} failures: {non_state_failures}"

    def test_dump_fast_dispatch(self):
        self.run_triage_script("dump_fast_dispatch.py")

    def test_dump_lightweight_asserts(self):
        result = self.run_triage_script("dump_lightweight_asserts.py")

        assert result is not None, "Expected non-None result from dump_lightweight_asserts.py"

        # Get expected results from configuration, skip detailed checks if not provided
        expected = self.expected_results.get("lightweight_asserts")
        if not expected:
            return  # No expected results configured, just verify it runs without failures

        if expected.get("compute_cores_hang"):
            # The whole compute pipeline stops on the kernel's ebreak, so every compute core of the
            # NEO that ran it should report -- and nothing else should.
            reported = {(check.neo_id, check.risc_name) for check in result}
            neo_ids = {neo_id for neo_id, _ in reported}
            assert len(neo_ids) == 1, f"Expected one NEO to have run the compute kernel, got {neo_ids}"
            expected_risc_names = compute_risc_names(result[0].location, next(iter(neo_ids)))
            risc_names = {name for _, name in reported}
            assert risc_names == expected_risc_names, f"Expected {expected_risc_names}, got {risc_names}"
            assert len(result) == len(reported), f"Expected one result per core, got {len(result)} for {reported}"

        for check in result:
            assert check.result is not None, f"Expected non-None result for {check.risc_name}"

            # Verify kernel name if specified
            expected_kernel_name = expected.get("kernel_name")
            if expected_kernel_name:
                assert (
                    check.result.kernel_name == expected_kernel_name
                ), f"{check.risc_name}: Expected kernel_name '{expected_kernel_name}', got '{check.result.kernel_name}'"

            # Verify callstack exists and has entries
            callstack = check.result.kernel_callstack_with_message.callstack.callstack
            assert callstack and len(callstack) > 0, f"{check.risc_name}: Callstack is empty"

            # Verify first callstack entry if specified
            first_entry = callstack[0]
            expected_file = expected.get("first_callstack_file")
            expected_line = expected.get("first_callstack_line")
            if expected_file or expected_line:
                assert (
                    first_entry.file_info is not None
                ), f"{check.risc_name}: Expected file_info on first callstack entry, got None"

            if expected_file:
                assert first_entry.file_info.file.endswith(
                    expected_file
                ), f"{check.risc_name}: Expected file ending with '{expected_file}', got '{first_entry.file_info.file}'"

            if expected_line:
                assert (
                    first_entry.file_info.line == expected_line
                ), f"{check.risc_name}: Expected line {expected_line}, got {first_entry.file_info.line}"

    def test_dump_configuration(self):
        result = self.run_triage_script("dump_configuration.py")
        assert result is not None, "Expected non-None result from dump_configuration.py"
        assert len(result) > 0, "Expected at least one configuration entry"

    def test_dump_running_operations(self):
        result = self.run_triage_script("dump_running_operations.py")

        expected = self.expected_results.get("running_operations")
        if not expected:
            return

        assert result is not None, "Expected non-None result from dump_running_operations.py"
        assert len(result) > 0, "Expected at least one running operation in dump_running_operations output"

        live_ops = [op for op in result if op.host_assigned_id]
        assert len(live_ops) > 0, (
            "Expected at least one running op with a non-zero host_assigned_id; "
            "got only background entries (op_id == 0)"
        )

        if expected.get("assert_no_na"):
            for op in live_ops:
                assert op.operation_name != "N/A", (
                    f"Op id {op.host_assigned_id}: operation_name resolved to N/A. "
                    f"Dispatcher host_assigned_id failed to lookup against "
                    f"Inspector getMeshWorkloadRuntimeEntries()."
                )
                assert op.operation_parameters != "N/A", (
                    f"Op id {op.host_assigned_id}: operation_parameters resolved to N/A. "
                    f"Op was named '{op.operation_name}' but params were empty in Inspector."
                )

        expected_name = expected.get("expected_op_name_contains")
        if expected_name:
            matching = [op for op in live_ops if expected_name in op.operation_name]
            assert len(matching) > 0, (
                f"No running op with name containing '{expected_name}'. "
                f"Got: {[op.operation_name for op in live_ops]}"
            )

    def test_dump_watcher_ringbuffer(self):
        self.run_triage_script("dump_watcher_ringbuffer.py")

    def test_dump_risc_debug_signals(self):
        self.run_triage_script("dump_risc_debug_signals.py")

    def test_dump_aggregated_callstacks(self):
        os.environ["TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS"] = "1"
        try:
            result = self.run_triage_script("dump_aggregated_callstacks.py")
            assert result is not None, "Expected non-None result from dump_aggregated_callstacks.py"

            # If we have valid kernel names, validate against expected results
            valid_rows = [row for row in result if row.kernel_name is not None]
            if len(valid_rows) > 0:
                expected = self.expected_results.get("lightweight_asserts")
                if expected and expected.get("kernel_name"):
                    expected_kernel_name = expected["kernel_name"]
                    expected_file = expected.get("first_callstack_file")

                    # Find aggregated row(s) with the expected kernel
                    matching_rows = [row for row in valid_rows if row.kernel_name == expected_kernel_name]
                    if len(matching_rows) > 0:
                        row = matching_rows[0]

                        # An aggregated row groups cores across locations, so it names a risc but no
                        # single core to look up. Assert what is left and is still true everywhere:
                        # the cores that stop on the kernel's ebreak are the compute ones.
                        if expected.get("compute_cores_hang"):
                            assert row.risc_name.startswith(
                                "trisc"
                            ), f"Expected a compute core to have hung, got '{row.risc_name}'"

                        # Validate callstack if expected
                        if expected_file and row.callstack:
                            callstack = row.callstack.callstack
                            assert len(callstack) > 0, "Expected non-empty callstack in aggregated row"
                            matching_entries = [
                                e for e in callstack if e.file_info and e.file_info.file.endswith(expected_file)
                            ]
                            assert len(matching_entries) > 0, (
                                f"Expected file '{expected_file}' not found in aggregated callstack. "
                                f"Callstack files: {[e.file_info.file if e.file_info else None for e in callstack]}"
                            )

        finally:
            os.environ.pop("TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS", None)

    # Running dump_callstacks with --full-callstack or --gdb-callstack breaks brisc so that it cannot be halted
    # and it affects other tests in the same test class, so we move it to be run last.
    def test_dump_callstacks(self):
        result = self.run_triage_script("dump_callstacks.py", argv=["--full-callstack"])

        assert result is not None, "Expected non-None result from dump_callstacks.py"

        # Get expected results from configuration
        expected = self.expected_results.get("callstacks")
        if not expected:
            # No expected results configured, just do basic validation
            return

        # Validate expected cores
        cores_to_check = expected.get("cores_to_check", {})
        location_to_check = expected.get("location_to_check")
        device_to_check = expected.get("device_to_check")

        # Filter results to only the expected cores and location
        filtered_results = result
        if location_to_check and device_to_check is not None:
            device = result[0].device_description.device  # Get device from first result
            expected_coord = OnChipCoordinate.create(location_to_check, device)
            filtered_results = [
                check
                for check in result
                if check.location == expected_coord and check.device_description.device.id == device_to_check
            ]

        results_by_risc = {check.risc_name: check for check in filtered_results if check.risc_name in cores_to_check}

        for risc_name, expected_data in cores_to_check.items():
            assert risc_name in results_by_risc, f"Expected {risc_name} in results, got {list(results_by_risc.keys())}"

            check = results_by_risc[risc_name]
            assert check.result is not None, f"Expected non-None result for {risc_name}"

            # Verify core is halted (stuck on ebreak)
            risc_debug = check.location.noc_block.get_risc_debug(risc_name, check.neo_id)
            assert risc_debug.is_halted(), f"{risc_name}: Core is not halted (not stuck on ebreak)"

            # Verify callstack
            callstack_with_message = check.result.kernel_callstack_with_message
            callstack = callstack_with_message.callstack
            assert len(callstack) > 0, f"{risc_name}: Callstack is empty"

            # Verify callstack contains expected file and line
            expected_file = expected_data.get("file")
            expected_line = expected_data.get("line")
            if expected_file:
                # Search through callstack to find the expected file/line
                matching_entries = [
                    entry for entry in callstack if entry.file_info and entry.file_info.file.endswith(expected_file)
                ]
                assert len(matching_entries) > 0, (
                    f"{risc_name}: Expected file '{expected_file}' not found in callstack. "
                    f"Callstack files: {[entry.file_info.file if entry.file_info else None for entry in callstack]}"
                )

                if expected_line is not None:
                    # Find entry with matching file and line
                    matching_entry = next(
                        (entry for entry in matching_entries if entry.file_info.line == expected_line), None
                    )
                    assert matching_entry is not None, (
                        f"{risc_name}: Expected file '{expected_file}' at line {expected_line} not found. "
                        f"Found {expected_file} at lines: {[entry.file_info.line for entry in matching_entries]}"
                    )

    def any_device_has_firmware(self) -> bool:
        return any(device_has_firmware(device) for device in self.exalens_context.devices.values())

    def assert_no_errors_or_none_in_result(self, result: list | None):
        assert result is not None, "Expected non-None result"
        assert len(result) > 0, "Expected at least one row"

        for check in result:
            device_id = check.device_description.device.id
            assert check.result is not None, f"No result for device {device_id}"

            for field in fields(check.result):
                value = str(getattr(check.result, field.name))
                assert (
                    "none" not in value.lower() and "error" not in value.lower()
                ), f"Device {device_id} field '{field.name}' is {value!r}"

    def run_triage_script(
        self,
        script_name: str,
        args: ScriptArguments = None,
        argv: list[str] = [],
        return_result: bool = True,
        assert_failure_checks: bool = True,
    ):
        global triage_home

        triage.CHECKS.clear()
        result = run_script(
            script_path=os.path.join(triage_home, script_name),
            args=args,
            context=self.exalens_context,
            argv=argv,
            return_result=return_result,
        )

        if assert_failure_checks:
            failures = logged_errors()
            assert len(failures) == 0, f"{script_name} failed with {len(failures)} failures: {failures}"

        return result


@pytest.mark.parametrize(
    "cause_hang_with_app",
    [
        (
            HANG_APP_MESH_SOCKET,
            [str(MESH_SOCKET_FIFO_SIZE)],
            {"min_devices": 2, "expect_running": True},
            15,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("cause_hang_with_app")
class TestMeshSocketTriage:
    exalens_context: Context

    def test_dump_mesh_sockets(self):
        global triage_home

        triage.CHECKS.clear()
        result = run_script(
            script_path=os.path.join(triage_home, "dump_mesh_sockets.py"),
            context=self.exalens_context,
            argv=[],
            return_result=True,
        )
        failures = logged_errors()
        assert not failures, f"dump_mesh_sockets.py failed with: {failures}"
        assert result is not None, "Expected socket rows while MeshSockets are wedged"

        rows = [check.result for check in result]
        device_of = {id(check.result): check.device_description.device.id for check in result}
        senders = [row for row in rows if row.role == "sender"]
        receivers = [row for row in rows if row.role == "receiver"]

        # One 1:1 socket pair each way, plus a fan-out sender feeding two receiver cores. The fan-out
        # sender is a single core, so it contributes one row per downstream.
        pair_senders = [row for row in senders if row.num_downstreams == 1]
        fanout_senders = [row for row in senders if row.num_downstreams == 2]
        assert len(pair_senders) == 2, f"Expected 2 paired sender rows, got {len(pair_senders)}"
        assert len(fanout_senders) == 2, f"Expected 2 fan-out sender rows, got {len(fanout_senders)}"
        assert len(receivers) == 4, f"Expected 4 receiver rows, got {len(receivers)}"
        assert len({device_of[id(r)] for r in pair_senders}) == 2, "Expected the paired senders on different devices"

        for row in rows:
            # A row only carries the columns that live in its own config buffer.
            if row.role == "sender":
                assert (row.sent_at_receiver, row.acked_at_receiver, row.read_ptr) == (None, None, None)
            else:
                assert (row.sent_at_sender, row.acked_at_sender, row.write_ptr) == (None, None, None)
                assert (row.downstream_config_addr, row.downstream, row.num_downstreams) == (None, None, None)
            assert row.fifo_size == MESH_SOCKET_FIFO_SIZE
            # Every core is on this host, so each names its peer by device id.
            assert row.peer.startswith("dev"), f"Expected a device id for a local peer, got {row.peer}"
            # Node is the core's own fabric node, so it agrees with the device the row came from.
            assert row.node == f"chip{device_of[id(row)]}/mesh0"

        # The fan-out rows come from one core, so they share its config buffer and differ
        # only in which downstream they describe.
        assert len({r.config_addr for r in fanout_senders}) == 1, "Fan-out rows should share one config buffer"
        assert len({str(r.location) for r in fanout_senders}) == 1, "Fan-out rows should share one core"
        assert sorted(r.downstream for r in fanout_senders) == [0, 1]
        assert len({r.peer for r in fanout_senders}) == 2, "Each downstream should name its own receiver core"

        # Downstream Addr is the documented join key: it names the peer receiver's config buffer. The
        # fan-out receivers are pages of one buffer, so they share an address and both match.
        receivers_by_config_addr: dict[int, list] = {}
        for row in receivers:
            receivers_by_config_addr.setdefault(row.config_addr, []).append(row)
        for snd in senders:
            matched = receivers_by_config_addr.get(snd.downstream_config_addr)
            assert matched, f"Sender at {snd.config_addr:#x} points at no receiver row"
            for rcv in matched:
                assert device_of[id(rcv)] != device_of[id(snd)], "Each socket should cross to the other device"
        assert len(receivers_by_config_addr[fanout_senders[0].downstream_config_addr]) == 2

        # Of the two 1:1 sockets, one is wedged with a full fifo and one never saw a byte.
        paired_receivers = [rcv for snd in pair_senders for rcv in receivers_by_config_addr[snd.downstream_config_addr]]
        sent = sorted(rcv.sent_at_receiver for rcv in paired_receivers)
        assert sent == [0, MESH_SOCKET_FIFO_SIZE], f"Expected a starved and a backpressured receiver, got {sent}"
