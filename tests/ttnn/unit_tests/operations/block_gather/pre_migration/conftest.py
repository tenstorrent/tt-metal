# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the block_gather pre-migration suite (this directory only)."""

import os
import sys

import pytest
import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # block_gather_pm_ref (pytest runs with --import-mode=importlib)


@pytest.fixture(scope="module")
def full_mesh():
    """The whole system as one mesh, tensors replicated: the golden suite's topology (block_gather itself needs no
    fabric, so none is configured)."""
    m = ttnn.open_mesh_device()
    yield m
    ttnn.close_mesh_device(m)
