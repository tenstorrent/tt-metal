# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test for 32x4 mesh device setup.
This test verifies that a 32x4 device mesh can be opened successfully with fabric configuration.
"""

import pytest
from loguru import logger

import ttnn


@pytest.mark.parametrize(
    "mesh_device",
    [(32, 4)],
    ids=["32x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_32x4_device_sanity(mesh_device):
    """
    Sanity test for 32x4 mesh device.
    Opens the device with linear fabric topology and performs synchronization.

    Args:
        mesh_device: 32x4 mesh device fixture with linear fabric
    """
    logger.info("Starting 32x4 device sanity test with linear fabric...")

    # Synchronize device operations
    ttnn.synchronize_device(mesh_device)
    logger.info("✓ Device synchronization completed")

    # Perform distributed synchronization
    ttnn.distributed_context_barrier()
    logger.info("✓ Distributed synchronization completed")

    logger.success("✓ 32x4 device sanity test executed successfully!")
    logger.info(f"✓ Mesh device opened with {mesh_device.get_num_devices()} devices")
    logger.info(f"✓ Mesh shape: {mesh_device.shape}")
    logger.info("✓ Linear fabric topology configured")
