# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for AceStep v1.5 demo tests."""

from __future__ import annotations

import os
import sys

import pytest
import torch

# This conftest lives at:
#   tt-metal/models/demos/ace_step_v1_5/tests/conftest.py
# We need the repo root `tt-metal/` on sys.path, and also `tt-metal/ttnn/`
# so `import ttnn` resolves to `tt-metal/ttnn/ttnn/__init__.py` (not the
# namespace package at `tt-metal/ttnn/`).
_TT_METAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_TTNN_ROOT = os.path.join(_TT_METAL_ROOT, "ttnn")
# NOTE: Do not add `tt-metal/tools` to sys.path here. Some environments contain an optional
# `tools/tracy` package that depends on extra plotting libs (e.g. seaborn). Importing TTNN
# should not require those extras for unit tests.
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
    ttnn = require_ttnn()
    dev = ttnn.open_device(device_id=0, trace_region_size=128 << 20)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def require_ttnn():
    # TTNN native extension may fail to initialize on hosts without a proper runtime.
    # Treat that as a skip for demo tests.
    return pytest.importorskip("ttnn", exc_type=ImportError)


@pytest.fixture(scope="session")
def mesh_device():
    """
    Single mesh for the whole test session.

    A function-scoped open/close loop exhausts Metal context slots and breaks
    subsequent ``open_mesh_device`` calls (invalid context_id / MAX_CONTEXT_COUNT).
    Remote-only meshes can also abort if many MeshDevice instances are torn down.
    """
    ttnn = require_ttnn()
    # Prefer a mesh device when supported.
    if hasattr(ttnn, "open_mesh_device") and hasattr(ttnn, "MeshShape") and os.environ.get("MESH_DEVICE"):
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        if hasattr(mesh, "enable_program_cache"):
            mesh.enable_program_cache()
        try:
            yield mesh
        finally:
            ttnn.close_mesh_device(mesh)
        return

    # Fallback: some TTNN builds only expose single-device APIs.
    if hasattr(ttnn, "open_device"):
        dev = ttnn.open_device(device_id=0, trace_region_size=128 << 20)
        if hasattr(dev, "enable_program_cache"):
            dev.enable_program_cache()
        try:
            yield dev
        finally:
            ttnn.close_device(dev)
        return

    pytest.skip("No TT device API available (missing open_mesh_device/open_device).")
