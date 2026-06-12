# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Voxtral test fixtures — runtime device selection for single-card workloads."""

from __future__ import annotations

import pytest
import ttnn
from loguru import logger

from models.experimental.voxtraltts.tests.common import (
    prepare_voxtral_open_mesh_kwargs,
    voxtral_single_device_mesh_shape,
)


@pytest.fixture(scope="function")
def voxtral_runtime_mesh_device(request, device_params):
    """Open a 1×1 mesh on whatever host is running (P150, QB rank, CI runner, …).

    Unlike the root ``mesh_device`` fixture with no ``request.param`` (which can
    open the full system mesh from ``SystemMeshDescriptor``), this always uses a
    single effective device — appropriate for audio tokenizer and other
    non-distributed Voxtral modules.

    ``device_params`` (including fabric/dispatch overrides from parametrization or
    CI) are merged with ``get_updated_device_params`` for the local architecture.

    Uses ``--device-id`` (default 0) like the root ``device`` fixture instead of
    ``get_pcie_device_ids()``, which scans every PCIe device and fails when any
    card is left in a hung NOC state after a killed run.
    """
    from conftest import first_available_tg_device, is_tg_cluster, reset_fabric, set_fabric

    device_id = request.config.getoption("device_id")
    if is_tg_cluster() and not device_id:
        device_id = first_available_tg_device()
    request.node.pci_ids = [ttnn.GetPCIeDeviceID(device_id)]

    mesh_shape = voxtral_single_device_mesh_shape()
    open_kwargs, fabric = prepare_voxtral_open_mesh_kwargs(device_params)
    set_fabric(
        fabric["fabric_config"],
        fabric["reliability_mode"],
        fabric["fabric_tensix_config"],
        fabric["fabric_manager"],
        fabric["fabric_router_config"],
    )

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        physical_device_ids=[device_id],
        **open_kwargs,
    )
    logger.info(
        "voxtral runtime mesh: shape={} arch={} num_devices={}",
        tuple(mesh_device.shape),
        ttnn.get_arch_name(),
        mesh_device.get_num_devices(),
    )

    try:
        yield mesh_device
    finally:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        reset_fabric(fabric["fabric_config"])
        del mesh_device
