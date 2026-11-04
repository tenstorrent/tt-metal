# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Opt-in standalone device fixtures; collection never opens hardware."""

import os
import pytest
from models.experimental.nllb.tests.process_runner import run_task


@pytest.fixture
def nllb_device_id():
    value = os.environ.get("NLLB_TEST_DEVICE")
    if value is None:
        pytest.skip("set NLLB_TEST_DEVICE to opt in to actual TT inference")
    return int(value)


@pytest.fixture
def nllb_component_runner(nllb_device_id):
    # A closed device can leave the process-global UMD chip lock alive. Each
    # component exits its process before the standalone CLI test starts.
    def run(component):
        result = run_task("component", {"component": component, "device": nllb_device_id}, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout)

    return run
