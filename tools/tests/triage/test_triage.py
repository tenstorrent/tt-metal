# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# You need to run these tests from python_env that is created with ./create_venv.sh
# You also need to install everything needed to run tt-triage.py in that environment
# Run manually ./tools/tt-triage.py --help to see if it works and install requirements

import os
import pytest
import subprocess
import time


def detect_metal_home():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="class")
def cause_hang_with_app(request):
    app, args, app_configuration, timeout = request.param
    build_dir = os.path.join(detect_metal_home(), "build")
    app_path_str = os.path.join(build_dir, app)
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
        if proc.returncode is None:
            raise RuntimeError("The application did not hang as expected.")
    else:
        time.sleep(timeout)
    request.cls.app_configuration = app_configuration
    try:
        yield
    finally:
        # Clean up the hung application
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        # TODO: Reset the device state after the hang
        # subprocess.run(["tt-smi", "-r"], check=True)


@pytest.mark.parametrize(
    "cause_hang_with_app",
    [
        (
            # Manual hang detection with timeout from outside
            "tools/tests/triage/hang_apps/add_2_integers_hang/triage_hang_app_add_2_integers_hang",
            [],
            {"option": "value"},
            3,
        ),
        (
            # Automatic hang detection with timeout inside the app and serialization of Inspector RPC data
            "tools/tests/triage/hang_apps/add_2_integers_hang/triage_hang_app_add_2_integers_hang",
            [],
            {
                "auto_timeout": True,
                "env": {
                    "TT_METAL_OPERATION_TIMEOUT_SECONDS": "0.5",
                },
            },
            5,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("cause_hang_with_app")
class TestTriage:
    app_configuration: dict

    def test_triage_help(self):
        metal_home = detect_metal_home()
        result = subprocess.run(
            [os.path.join(metal_home, "tools", "tt-triage.py"), "--help"],
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
        metal_home = detect_metal_home()
        result = subprocess.run(
            [os.path.join(metal_home, "tools", "tt-triage.py")],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert len(result.stderr) == 0
