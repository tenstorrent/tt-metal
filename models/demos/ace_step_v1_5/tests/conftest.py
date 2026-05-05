# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for AceStep v1.5 demo tests."""

from __future__ import annotations

import os
import sys

import pytest
import torch

import ttnn

_TT_METAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TTNN_ROOT = os.path.join(_TT_METAL_ROOT, "ttnn")
for _p in (_TT_METAL_ROOT, _TTNN_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def torch_seed():
    torch.manual_seed(42)
    yield 42
    torch.manual_seed(42)


@pytest.fixture(scope="session")
def device():
    dev = ttnn.open_device(device_id=0, trace_region_size=128 << 20)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


import os

import pytest


def require_ttnn():
    return pytest.importorskip("ttnn")


@pytest.fixture(scope="function")
def mesh_device():
    ttnn = require_ttnn()
    if not os.environ.get("MESH_DEVICE"):
        pytest.skip("Requires TT device (set MESH_DEVICE)")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
