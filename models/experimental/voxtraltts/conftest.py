# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral pytest fixtures — P150 single card or BH QB2 1×4 host mesh (+ optional 1×4 compute)."""

from __future__ import annotations

import os

import pytest

from models.experimental.voxtraltts.demo.decode_trace_2cq import (
    num_command_queues_for_decode,
    reset_decode_trace_config,
)
from models.experimental.voxtraltts.utils.common import (
    close_voxtral_runtime_mesh,
    open_voxtral_runtime_mesh,
    voxtral_resolve_physical_device_id,
)
from tests.tests_common.cache_entries_counter import CacheEntriesCounter


def voxtral_trace_device_params() -> dict[str, int]:
    """Device open kwargs for traced decode (2 CQ + trace region)."""
    return {
        "trace_region_size": int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", str(200_000_000))),
        "num_command_queues": num_command_queues_for_decode(),
    }


def _voxtral_merge_device_params(device_params: dict | None) -> dict:
    """Apply decode-trace defaults unless a test overrides them explicitly."""
    params = dict(device_params or {})
    trace_defaults = voxtral_trace_device_params()
    for key, value in trace_defaults.items():
        params.setdefault(key, value)
    return params


@pytest.fixture(autouse=True)
def _reset_voxtral_decode_trace_config():
    reset_decode_trace_config()
    yield
    reset_decode_trace_config()


@pytest.fixture(scope="function")
def device(request, device_params):
    """P150: single device. QB2: 1×4 host mesh; compute is 1×1 submesh unless ``MESH_DEVICE=P150x4``."""
    physical_device_id = voxtral_resolve_physical_device_id(request.config.getoption("device_id"))
    runtime = open_voxtral_runtime_mesh(_voxtral_merge_device_params(device_params), device_id=physical_device_id)
    request.node.pci_ids = runtime.physical_device_ids
    runtime.compute_device.cache_entries_counter = CacheEntriesCounter(runtime.compute_device)

    try:
        yield runtime.compute_device
    finally:
        close_voxtral_runtime_mesh(runtime)
        del runtime


@pytest.fixture(scope="function")
def voxtral_runtime(request, device_params):
    """Full :class:`VoxtralRuntimeMesh` (host + compute topology) for diagnostics."""
    physical_device_id = voxtral_resolve_physical_device_id(request.config.getoption("device_id"))
    runtime = open_voxtral_runtime_mesh(_voxtral_merge_device_params(device_params), device_id=physical_device_id)
    request.node.pci_ids = runtime.physical_device_ids
    runtime.compute_device.cache_entries_counter = CacheEntriesCounter(runtime.compute_device)
    try:
        yield runtime
    finally:
        close_voxtral_runtime_mesh(runtime)
        del runtime


@pytest.fixture(scope="function")
def voxtral_runtime_mesh_device(device):
    """Alias kept for tests that name this fixture explicitly."""
    yield device
