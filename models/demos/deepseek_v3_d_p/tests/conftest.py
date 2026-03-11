# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for deepseek_v3_d_p tests.

Provides the `requires_mesh_topology` marker for skipping tests based on
device count and topology constraints for Blackhole vs Wormhole architectures.
"""

import pytest

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_mesh_topology(mesh_shape, topology): mark test to run only on compatible "
        "device/topology combinations. mesh_shape is (rows, cols) tuple, topology is 'ring' or 'linear'. "
        "Skips automatically based on available devices and arch constraints.",
    )


def pytest_collection_modifyitems(config, items):
    """
    Skip tests based on mesh/topology requirements at collection time.

    Hardware constraints:
    - Blackhole: Only supports 4-device configs (linear-4, ring-4)
    - Wormhole: Ring topology only works with 8 devices (ring-8)
    """
    num_devices = ttnn.get_num_devices()

    for item in items:
        marker = item.get_closest_marker("requires_mesh_topology")
        if not marker:
            continue

        # Extract marker arguments
        mesh_shape = marker.kwargs.get("mesh_shape") or (marker.args[0] if marker.args else None)
        topology = marker.kwargs.get("topology") or (marker.args[1] if len(marker.args) > 1 else None)

        if mesh_shape is None or topology is None:
            continue

        devices_needed = mesh_shape[0] * mesh_shape[1]
        is_ring = topology == "ring"

        skip_reason = None

        # Check device count first
        if devices_needed > num_devices:
            skip_reason = f"Requires {devices_needed} devices, only {num_devices} available"

        # Architecture-specific constraints
        elif is_blackhole():
            # BH: only supports 4-device configs
            if devices_needed != 4:
                skip_reason = f"Blackhole only supports 4-device mesh configs (requested {devices_needed})"

        elif is_wormhole_b0():
            # WH: ring topology only works with 8 devices
            if is_ring and devices_needed != 8:
                skip_reason = f"Wormhole ring topology only works with 8 devices (requested ring-{devices_needed})"

        if skip_reason:
            item.add_marker(pytest.mark.skip(reason=skip_reason))
