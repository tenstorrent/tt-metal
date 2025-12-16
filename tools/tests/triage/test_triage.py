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


from triage import run_script, FAILURE_CHECKS
from ttexalens.context import Context
from ttexalens.tt_exalens_init import init_ttexalens


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
            "tools/tests/triage/hang_apps/add_2_integers_hang/triage_hang_app_add_2_integers_hang",
            [],
            {},
            10,
        ),
        (
            # Automatic hang detection with timeout inside the app and serialization of Inspector RPC data
            "tools/tests/triage/hang_apps/add_2_integers_hang/triage_hang_app_add_2_integers_hang",
            [],
            {
                "auto_timeout": True,
                "env": {
                    "TT_METAL_OPERATION_TIMEOUT_SECONDS": "0.5",
                    "TT_METAL_LOGS_PATH": "/tmp/tt-metal/triage-test",
                },
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

    def test_check_binary_integrity(self):
        global triage_home
        global FAILURE_CHECKS

        FAILURE_CHECKS.clear()
        result = run_script(
            script_path=os.path.join(triage_home, "check_binary_integrity.py"),
            args=None,
            context=self.exalens_context,
            argv=[],
            return_result=True,
        )
        assert (
            len(FAILURE_CHECKS) == 0
        ), f"Binary integrity check failed with {len(FAILURE_CHECKS)} failures: {FAILURE_CHECKS}"

    def test_dump_fast_dispatch(self):
        global triage_home
        global FAILURE_CHECKS

        FAILURE_CHECKS.clear()
        result = run_script(
            script_path=os.path.join(triage_home, "dump_fast_dispatch.py"),
            args=None,
            context=self.exalens_context,
            argv=[],
            return_result=True,
        )
        assert (
            len(FAILURE_CHECKS) == 0
        ), f"Dump fast dispatch check failed with {len(FAILURE_CHECKS)} failures: {FAILURE_CHECKS}"

    def test_check_noc_status(self):
        global triage_home
        global FAILURE_CHECKS

        FAILURE_CHECKS.clear()
        result = run_script(
            script_path=os.path.join(triage_home, "check_noc_status.py"),
            args=None,
            context=self.exalens_context,
            argv=[],
            return_result=True,
        )
        # Some mismatches may occur on unused cores.
        non_state_failures = [failure for failure in FAILURE_CHECKS if "Mismatched state" not in failure]
        assert (
            len(non_state_failures) == 0
        ), f"Check NOC status check failed with {len(non_state_failures)} failures: {non_state_failures}"

    def test_check_arc(self):
        global triage_home
        global FAILURE_CHECKS

        FAILURE_CHECKS.clear()
        result = run_script(
            script_path=os.path.join(triage_home, "check_arc.py"),
            args=None,
            context=self.exalens_context,
            argv=[],
            return_result=True,
        )

        assert len(FAILURE_CHECKS) == 0, f"Arc check failed with {len(FAILURE_CHECKS)} failures: {FAILURE_CHECKS}"
        for check in result:
            assert (
                check.result.location == check.device_description.device.arc_block.location
            ), f"Incorrect ARC location: {check.result.location}"
            assert 0 < check.result.clock_mhz < 10000, f"Invalid ARC clock: {check.result.clock_mhz}"
            assert (
                timedelta(seconds=0) < check.result.uptime < timedelta(days=8 * 365)
            ), f"Invalid ARC uptime: {check.result.uptime}"

    def test_check_core_magic(self):
        global triage_home
        global FAILURE_CHECKS

        FAILURE_CHECKS.clear()
        run_script(
            script_path=os.path.join(triage_home, "check_core_magic.py"),
            args=None,
            context=self.exalens_context,
            argv=[],
            return_result=True,
        )
        assert (
            len(FAILURE_CHECKS) == 0
        ), f"Core magic check failed with {len(FAILURE_CHECKS)} failures: {FAILURE_CHECKS}"
