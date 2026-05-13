# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device fixtures for Kokoro experimental TTNN tests (1×1 mesh)."""

from __future__ import annotations

import pytest

import ttnn


@pytest.fixture(scope="function")
def mesh_device(request):
    """Match ``@pytest.mark.parametrize(\"mesh_device\", [1], indirect=True)`` from ported tests."""
    param = getattr(request, "param", 1)
    if isinstance(param, tuple) and len(param) == 2:
        mesh_shape = ttnn.MeshShape(int(param[0]), int(param[1]))
    elif isinstance(param, int) and param > 1:
        mesh_shape = ttnn.MeshShape(1, param)
    else:
        mesh_shape = ttnn.MeshShape(1, 1)

    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, l1_small_size=24576)
    yield mesh_device
    ttnn.close_mesh_device(mesh_device)


@pytest.fixture(scope="function")
def device():
    """Single-device alias for tests that open a ``ttnn.Device`` directly (predictor, text encoder).

    Mirrors the root-conftest ``device`` fixture so kokoro tests run under
    ``--confcutdir=models/experimental/kokoro`` without pulling in the rest of the repo's pytest setup.
    """
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)
