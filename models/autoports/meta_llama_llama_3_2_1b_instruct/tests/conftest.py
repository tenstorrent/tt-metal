# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import ttnn


def pytest_configure(config):
    config.addinivalue_line("markers", "real_weights: uses locally cached HuggingFace checkpoint weights")
    config.addinivalue_line("markers", "long_context: exercises long paged-prefill sequence lengths")
    config.addinivalue_line("markers", "perf_artifact: emits Tracy signposts for tt-perf-report artifacts")


@pytest.fixture
def device_params(request):
    return getattr(request, "param", {})


@pytest.fixture
def mesh_device(request, device_params):
    mesh_shape = getattr(request, "param", (1, 1))
    updated_device_params = dict(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    if fabric_config is not None:
        ttnn.set_fabric_config(fabric_config)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mesh_shape), **updated_device_params)
    try:
        yield device
    finally:
        ttnn.close_mesh_device(device)
        if fabric_config is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
