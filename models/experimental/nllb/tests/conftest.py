# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Opt-in standalone device fixtures; collection never opens hardware."""

import os
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.fixture
def nllb_device_id():
    value = os.environ.get("NLLB_TEST_DEVICE")
    if value is None:
        pytest.skip("set NLLB_TEST_DEVICE to opt in to actual TT inference")
    return int(value)


@pytest.fixture
def nllb_component_runner(nllb_device_id, tmp_path):
    # A closed device can leave the process-global UMD chip lock alive. Each
    # component exits its process before the standalone CLI test starts.
    def run(path):
        probe = """import runpy, sys, torch, ttnn
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
check = runpy.run_path(sys.argv[1])['check']
from models.experimental.nllb.tt import runtime_setup
runtime_setup.configure_tracking()
with runtime_setup.RuntimeOwner() as owner:
    device = owner.open(int(sys.argv[2]))
    check(device)
"""
        result = subprocess.run(
            [sys.executable, "-c", probe, str(Path(path).resolve()), str(nllb_device_id)],
            cwd=Path(__file__).resolve().parents[4],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout)

    return run
