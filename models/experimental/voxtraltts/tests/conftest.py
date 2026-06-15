# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Voxtral test fixtures — P150 single card or BH QB2 1×4 host mesh + 1×1 compute submesh."""

from __future__ import annotations

import pytest

from models.experimental.voxtraltts.tests.common import (
    close_voxtral_runtime_mesh,
    open_voxtral_runtime_mesh,
    voxtral_resolve_physical_device_id,
)
from tests.tests_common.cache_entries_counter import CacheEntriesCounter


@pytest.fixture(scope="function")
def device(request, device_params):
    """P150: single device. QB2: 1×4 host mesh with 1×1 compute submesh (audio-safe)."""
    physical_device_id = voxtral_resolve_physical_device_id(request.config.getoption("device_id"))
    runtime = open_voxtral_runtime_mesh(device_params, device_id=physical_device_id)
    request.node.pci_ids = runtime.physical_device_ids
    runtime.compute_device.cache_entries_counter = CacheEntriesCounter(runtime.compute_device)

    try:
        yield runtime.compute_device
    finally:
        close_voxtral_runtime_mesh(runtime)
        del runtime


@pytest.fixture(scope="function")
def voxtral_runtime_mesh_device(device):
    """Alias kept for tests that name this fixture explicitly."""
    yield device
