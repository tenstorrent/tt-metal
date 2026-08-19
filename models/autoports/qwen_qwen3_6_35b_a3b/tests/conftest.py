# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

import ttnn


def pytest_configure(config):
    config.addinivalue_line("markers", "real_weights: load checkpoint tensors for Qwen3.6 decoder parity")
    config.addinivalue_line("markers", "perf: signposted warmed performance capture for decoder evidence")
    config.addinivalue_line("markers", "context: larger context/sequence probes for decoder capability evidence")


@pytest.fixture(scope="function")
def device_params(request):
    return getattr(request, "param", {})


@pytest.fixture(scope="function")
def device(device_params):
    device_id = int(os.environ.get("TT_DEVICE_ID", "0"))
    original_default_device = ttnn.GetDefaultDevice()
    device = ttnn.CreateDevice(device_id=device_id, **device_params)
    ttnn.SetDefaultDevice(device)
    try:
        yield device
    finally:
        ttnn.SetDefaultDevice(original_default_device)
        ttnn.close_device(device)


@pytest.fixture(scope="function")
def mesh_device_params(request):
    return getattr(request, "param", {})


@pytest.fixture(scope="function")
def mesh_device(mesh_device_params):
    params = dict(mesh_device_params)
    fabric_config = params.pop("fabric_config", ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_shape = params.pop("mesh_shape", (2, 2))
    if fabric_config is not None:
        ttnn.set_fabric_config(fabric_config)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape), **params)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        if fabric_config is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
