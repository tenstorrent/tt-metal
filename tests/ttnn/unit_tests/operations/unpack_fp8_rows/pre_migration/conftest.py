# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the unpack_fp8_rows pre-migration suite: one mesh of the whole box (each chip decodes its own rows, as
the golden suite runs the op), opened once per session with the program cache enabled."""

import os
import sys

import pytest
import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # unpack_fp8_rows_pre_migration_helpers (pytest runs with --import-mode=importlib)


@pytest.fixture(scope="session")
def unpack_mesh():
    mesh = ttnn.open_mesh_device()
    mesh.enable_program_cache()
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.fixture
def fresh_cache(unpack_mesh):
    """A known program-cache state: empty and enabled."""
    ttnn.synchronize_device(unpack_mesh)
    unpack_mesh.clear_program_cache()
    unpack_mesh.enable_program_cache()
    assert unpack_mesh.num_program_cache_entries() == 0
    return unpack_mesh
