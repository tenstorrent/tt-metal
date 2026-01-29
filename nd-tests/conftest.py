# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for nd-tests.

This conftest provides the device fixture used by tests.
"""

import pytest
import ttnn


@pytest.fixture(scope="function")
def device_params(request):
    """
    Fixture to capture device parameters from parametrize decorator.

    Used with indirect=True in @pytest.mark.parametrize to pass
    parameters to the device fixture.

    Returns empty dict if not parametrized.
    """
    if hasattr(request, "param"):
        return request.param
    return {}


@pytest.fixture(scope="function")
def device(device_params):
    """
    Device fixture that handles device initialization and cleanup.

    Supports parametrization via device_params fixture for custom device config.
    """
    if device_params is None:
        device_params = {}

    device_id = device_params.get("device_id", 0)
    num_command_queues = device_params.get("num_command_queues", 1)
    l1_small_size = device_params.get("l1_small_size", 32768)
    trace_region_size = device_params.get("trace_region_size", 0)

    device = ttnn.open_device(
        device_id=device_id,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        num_command_queues=num_command_queues,
    )

    yield device

    ttnn.close_device(device)
