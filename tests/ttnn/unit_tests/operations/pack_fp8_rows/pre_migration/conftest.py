# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the pack_fp8_rows pre-migration suite (this directory only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # pack_fp8_rows_pre_migration_helpers (pytest runs with --import-mode=importlib)


@pytest.fixture
def program_cache(device):
    """The device's program cache, enabled and emptied: every test that counts entries starts from a known state."""
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    device.clear_program_cache()
    assert device.num_program_cache_entries() == 0
    return device


@pytest.fixture
def entries(device):
    """entries() -> the device's program cache entry count."""
    return device.num_program_cache_entries


@pytest.fixture(scope="module")
def golden_mesh():
    """The golden suite's own mesh: the whole box, FABRIC_2D, l1_small_size 24576."""
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
