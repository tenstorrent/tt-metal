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


def require_ttnn():
    return pytest.importorskip("ttnn")


@pytest.fixture(scope="session")
def mesh_device():
    """
    Single mesh for the whole test session.

    A function-scoped open/close loop exhausts Metal context slots and breaks
    subsequent ``open_mesh_device`` calls (invalid context_id / MAX_CONTEXT_COUNT).
    Remote-only meshes can also abort if many MeshDevice instances are torn down.
    """
    ttnn = require_ttnn()
    if not os.environ.get("MESH_DEVICE"):
        pytest.skip("Requires TT device (set MESH_DEVICE)")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    if hasattr(mesh, "enable_program_cache"):
        mesh.enable_program_cache()
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
