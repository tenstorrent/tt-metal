# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest for binary_ng benchmarks.

Provides module-scoped device fixture that reads profiler data after all tests complete.
"""

import pytest
import os
import sys
from pathlib import Path
import ttnn

# Import helper functions from root conftest and common scripts
# These are available in pytest's import context
import importlib.util

# Path: tests/ttnn/benchmarks/binary_ng/conftest.py -> workspace root
root_conftest_path = Path(__file__).parent.parent.parent.parent.parent / "conftest.py"
spec = importlib.util.spec_from_file_location("root_conftest", root_conftest_path)
root_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_conftest)

from tests.scripts.common import get_updated_device_params


@pytest.fixture(scope="module")
def device_with_profiling(request):
    """
    Module-scoped device fixture that reads profiler data after all tests complete.

    This ensures ReadDeviceProfiler() is called while the device is still open,
    which is required for reliable profiler data collection.

    Uses the same device_id logic as the global device fixture.
    """
    device_id = request.config.getoption("device_id")

    # When initializing a single device on a TG system, we want to
    # target the first user exposed device, not device 0
    if root_conftest.is_tg_cluster() and not device_id:
        device_id = root_conftest.first_available_tg_device()

    device_params = getattr(request, "param", {})
    updated_device_params = get_updated_device_params(device_params)

    device = ttnn.CreateDevice(device_id=device_id, **updated_device_params)
    ttnn.SetDefaultDevice(device)

    yield device

    # After all tests in this module complete, read profiler data if enabled
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        try:
            print(f"\n[PROFILER] Synchronizing device before reading profiler data...")
            ttnn.synchronize_device(device)  # Ensure all operations complete before reading profiler
            print(f"[PROFILER] Reading device profiler data after all tests complete...")
            ttnn.ReadDeviceProfiler(device)
            print(f"[PROFILER] Successfully read profiler data")
        except Exception as e:
            # Log but don't fail - profiler data may not be critical for test pass/fail
            print(f"[PROFILER] Warning: Failed to read profiler data: {e}")

    ttnn.close_device(device)
