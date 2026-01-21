# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# You need to run these tests from python_env that is created with ./create_venv.sh
# You also need to install everything needed to run tt-triage.py in that environment
# Run manually ./tools/tt-triage.py --help to see if it works and install requirements

from datetime import timedelta
import os
import sys
import pytest
import subprocess
import time


metal_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
triage_script = os.path.join(metal_home, "tools", "tt-triage.py")
triage_home = os.path.join(metal_home, "tools", "triage")


# Add triage tools directory to Python path
sys.path.insert(0, triage_home)


import triage
from triage import run_script, FAILURE_CHECKS, ScriptArguments
from ttexalens.context import Context
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.coordinate import OnChipCoordinate


triage.progress_disabled = True  # Disable progress bars for tests

# Mapping of hang application paths to their expected test results
HANG_APP_ADD_2_INTEGERS = "tools/tests/triage/hang_apps/add_2_integers_hang/triage_hang_app_add_2_integers_hang"
HANG_APP_EXPECTED_RESULTS = {
    HANG_APP_ADD_2_INTEGERS: {
        "lightweight_asserts": {
            "kernel_name": "add_2_tiles_hang",
            "risc_names": {"trisc0", "trisc1", "trisc2"},
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
}


def print_process_output(proc):
    stdout, stderr = proc.communicate(input=None, timeout=0)
    print("\n=== Process stdout ===")
    print(stdout.decode("utf-8") if stdout else "(empty)")
    print("\n=== Process stderr ===")
    print(stderr.decode("utf-8") if stderr else "(empty)")


@pytest.fixture(scope="class")
def cause_hang_with_app(request):
    global metal_home

    app, args, app_configuration, timeout = request.param
    build_dir = os.path.join(metal_home, "build")
    app_path_str = os.path.join(build_dir, app)
    os.environ.pop("TT_METAL_LOGS_PATH", None)
    proc = subprocess.Popen(
        [app_path_str] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **app_configuration.get("env", {})},
    )
    auto_timeout = app_configuration.get("auto_timeout", False)
    if auto_timeout:
        # Wait for the application to hang itself
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

        # Check if the process has exited
        if proc.returncode != 0:
            # Print process output for debugging
            print("The application did not hang as expected.")
            print_process_output(proc)
            raise RuntimeError("The application did not hang as expected.")
    else:
        time.sleep(timeout)

    request.cls.app_configuration = app_configuration
    request.cls.expected_results = app_configuration.get("expected_results", {})
    request.cls.exalens_context = init_ttexalens()
    if app_configuration.get("env", {}).get("TT_METAL_LOGS_PATH"):
        metal_logs_path = app_configuration["env"]["TT_METAL_LOGS_PATH"]
        os.environ["TT_METAL_LOGS_PATH"] = metal_logs_path
    try:
        yield
    finally:
        # Clean up the hung application
        proc.terminate()
        try:
            proc.wait(timeout=5)
            # Pytest will only display this if test fails
            print_process_output(proc)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

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
            # Automatic hang detection with timeout inside the app and serialization of Inspector RPC data
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
            20,
        ),
        (
            # Automatic hang detection with timeout inside the app and serialization of Inspector RPC data
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
            20,
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

    def test_triage_initialize_with_noc1(self):
        global triage_script

        result = subprocess.run(
            [triage_script, "--initialize-with-noc1", "--run=test_output"],
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

    def test_check_binary_integrity(self):
        self.run_triage_script("check_binary_integrity.py")

    def test_check_cb_inactive(self):
        self.run_triage_script("check_cb_inactive.py")

    def test_check_core_magic(self):
        self.run_triage_script("check_core_magic.py")

    def test_check_eth_status(self):
        self.run_triage_script("check_eth_status.py")

    def test_check_noc_locations(self):
        self.run_triage_script("check_noc_locations.py")

    def test_check_noc_status(self):
        self.run_triage_script("check_noc_status.py", assert_failure_checks=False)

        global FAILURE_CHECKS

        # Some mismatches may occur on unused cores.
        non_state_failures = [failure for failure in FAILURE_CHECKS if "Mismatched state" not in failure]
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

        expected_risc_names = expected.get("risc_names")
        if expected_risc_names:
            assert len(result) == len(
                expected_risc_names
            ), f"Expected {len(expected_risc_names)} risc results, got {len(result)}"
            risc_names = {check.risc_name for check in result}
            assert risc_names == expected_risc_names, f"Expected {expected_risc_names}, got {risc_names}"

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
            if expected_file:
                assert first_entry.file.endswith(
                    expected_file
                ), f"{check.risc_name}: Expected file ending with '{expected_file}', got '{first_entry.file}'"

            expected_line = expected.get("first_callstack_line")
            if expected_line:
                assert (
                    first_entry.line == expected_line
                ), f"{check.risc_name}: Expected line {expected_line}, got {first_entry.line}"

    def test_dump_running_operations(self):
        self.run_triage_script("dump_running_operations.py")

    def test_dump_watcher_ringbuffer(self):
        self.run_triage_script("dump_watcher_ringbuffer.py")

    def test_dump_risc_debug_signals(self):
        self.run_triage_script("dump_risc_debug_signals.py")

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
            risc_debug = check.location.noc_block.get_risc_debug(risc_name)
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
                matching_entries = [entry for entry in callstack if entry.file.endswith(expected_file)]
                assert len(matching_entries) > 0, (
                    f"{risc_name}: Expected file '{expected_file}' not found in callstack. "
                    f"Callstack files: {[entry.file for entry in callstack]}"
                )

                if expected_line is not None:
                    # Find entry with matching file and line
                    matching_entry = next((entry for entry in matching_entries if entry.line == expected_line), None)
                    assert matching_entry is not None, (
                        f"{risc_name}: Expected file '{expected_file}' at line {expected_line} not found. "
                        f"Found {expected_file} at lines: {[entry.line for entry in matching_entries]}"
                    )

    def run_triage_script(
        self,
        script_name: str,
        args: ScriptArguments = None,
        argv: list[str] = [],
        return_result: bool = True,
        assert_failure_checks: bool = True,
    ):
        global triage_home
        global FAILURE_CHECKS

        FAILURE_CHECKS.clear()
        result = run_script(
            script_path=os.path.join(triage_home, script_name),
            args=args,
            context=self.exalens_context,
            argv=argv,
            return_result=return_result,
        )

        if assert_failure_checks:
            assert (
                len(FAILURE_CHECKS) == 0
            ), f"{script_name} failed with {len(FAILURE_CHECKS)} failures: {FAILURE_CHECKS}"

        return result
