# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Custom conftest to allow selecting which device is the coordinator.

Usage:
    # Use device 3 as coordinator:
    export TT_COORDINATOR_DEVICE_ID=3
    pytest test_my_model.py -k "batch-32"

    # Your monitor will now show:
    # Device 3: L1 = 3.17 MB  ← NEW coordinator!
    # Devices 0,1,2,4,5,6,7: L1 = 73 KB
"""

import os

import pytest

import ttnn


@pytest.fixture
def device_params(request, galaxy_type):
    """Enhanced device_params that supports custom coordinator selection."""
    params = getattr(request, "param", {}).copy()

    mesh_device = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
        os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
    )
    is_single_device = (mesh_device == (1, 1)) if isinstance(mesh_device, tuple) else (mesh_device == 1)

    if "fabric_config" in params:
        if is_single_device:
            params["fabric_config"] = None
        elif params["fabric_config"] == True:
            params["fabric_config"] = (
                ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
            )

    # NEW: Support custom coordinator selection
    coordinator_id = os.environ.get("TT_COORDINATOR_DEVICE_ID")
    if coordinator_id is not None:
        coordinator_id = int(coordinator_id)
        all_device_ids = ttnn.get_device_ids()

        if coordinator_id not in all_device_ids:
            raise ValueError(f"TT_COORDINATOR_DEVICE_ID={coordinator_id} not in available devices: {all_device_ids}")

        # Reorder devices: put coordinator first
        physical_device_ids = [coordinator_id] + [d for d in all_device_ids if d != coordinator_id]
        params["physical_device_ids"] = physical_device_ids

        print(f"\n{'='*80}")
        print(f"CUSTOM COORDINATOR SELECTED: Device {coordinator_id}")
        print(f"Physical device order: {physical_device_ids}")
        print(f"  Device {coordinator_id} → Logical device 0 (coordinator)")
        for i, phys_id in enumerate(physical_device_ids[1:], start=1):
            print(f"  Device {phys_id} → Logical device {i} (worker)")
        print(f"{'='*80}\n")

    return params


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    """
    Enhanced mesh_device fixture that supports custom coordinator via physical_device_ids.
    """
    mesh_device_shape = request.param
    mesh_shape = None

    if isinstance(mesh_device_shape, tuple):
        mesh_shape = ttnn.MeshShape(*mesh_device_shape)

    physical_device_ids = device_params.pop("physical_device_ids", [])

    # Open mesh with custom device ordering
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        physical_device_ids=physical_device_ids,  # ← Custom ordering!
        **device_params,
    )

    yield mesh_device

    ttnn.close_mesh_device(mesh_device)
    del mesh_device


@pytest.fixture(scope="session")
def galaxy_type():
    """
    Determines the galaxy type based on environment variables or defaults to 6U.
    """
    galaxy_type_env = os.getenv("GALAXY_TYPE", "6U")
    return galaxy_type_env
