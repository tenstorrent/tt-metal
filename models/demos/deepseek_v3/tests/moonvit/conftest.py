# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Test fixtures for MoonViT submodule PCC tests.

The `mesh_device` fixture is provided by tt-metal's root conftest
(`/conftest.py` line 526). Tests should parametrize it `indirect=True`
with a `(rows, cols)` tuple matching the target topology — for the
v1 single-device prototype this is `(1, 1)`.

This conftest adds:
  - `ensure_gc`: autouse GC sweep between tests (avoids stale tensors).
  - `model_args`: a session-scoped MoonViTModelArgs bound to the mesh
    device. Constructing this lazy-loads the HF config.
"""
from __future__ import annotations

import gc

import pytest

from models.demos.deepseek_v3.tt.moonvit.model_config import MoonViTModelArgs


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


@pytest.fixture(scope="function")
def model_args(mesh_device):
    """MoonViTModelArgs bound to the test's mesh_device.

    Tests that need the HF reference modules use this fixture, then
    call `model_args.reference_*()` to get the comparison target.
    """
    import ttnn

    return MoonViTModelArgs(mesh_device=mesh_device, dtype=ttnn.bfloat16)
