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
from loguru import logger
from tests.scripts.common import get_updated_device_params

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
            # BH: only supports all available devices configs
            if devices_needed != num_devices:
                skip_reason = f"Blackhole only supports {num_devices}-device mesh configs (requested {devices_needed})"

        elif is_wormhole_b0():
            # WH: ring topology only works with 8 devices
            if is_ring and devices_needed != 8:
                skip_reason = f"Wormhole ring topology only works with 8 devices (requested ring-{devices_needed})"

        if skip_reason:
            item.add_marker(pytest.mark.skip(reason=skip_reason))


CLUSTER_TO_MESH_SHAPE = {
    ttnn.cluster.ClusterType.BLACKHOLE_GALAXY: ttnn.MeshShape(8, 4),
    ttnn.cluster.ClusterType.P150_X8: ttnn.MeshShape(4, 2),
    ttnn.cluster.ClusterType.P150_X4: ttnn.MeshShape(2, 2),
}


def _open_mesh(device_params, mesh_shape=None):
    """Open a 2D mesh device, auto-detecting shape from cluster type if not specified."""
    updated_params = get_updated_device_params(device_params or {})
    fabric_config = updated_params.pop("fabric_config", None)
    if fabric_config and fabric_config != ttnn.FabricConfig.DISABLED:
        ttnn.set_fabric_config(fabric_config)

    if mesh_shape is None:
        cluster_type = ttnn.cluster.get_cluster_type()
        if cluster_type not in CLUSTER_TO_MESH_SHAPE:
            raise ValueError(
                f"Unsupported cluster type {cluster_type}, " f"expected one of {list(CLUSTER_TO_MESH_SHAPE.keys())}"
            )
        mesh_shape = CLUSTER_TO_MESH_SHAPE[cluster_type]

    updated_params.setdefault("mesh_shape", mesh_shape)
    mesh = ttnn.open_mesh_device(**updated_params)
    logger.debug(f"Opened mesh device: shape={mesh.shape}, devices={mesh.get_num_devices()}")
    return mesh, fabric_config


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    """
    2D mesh device fixture for Blackhole-based DeepSeek prefill tests.

    Supports indirect parametrization with a (rows, cols) tuple to override
    the auto-detected mesh shape::

        @pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True)
        def test_something(mesh_device): ...

    Without indirect params, auto-detects from cluster type:
        BLACKHOLE_GALAXY -> (8, 4)
        P150_X8          -> (4, 2)
        P150_X4          -> (2, 2)
    """
    shape_override = getattr(request, "param", None)
    if shape_override is not None:
        mesh_shape = ttnn.MeshShape(*shape_override)
    else:
        mesh_shape = None

    mesh, fabric_config = _open_mesh(device_params, mesh_shape)
    yield mesh

    ttnn.close_mesh_device(mesh)
    if fabric_config and fabric_config != ttnn.FabricConfig.DISABLED:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
