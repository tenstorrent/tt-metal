# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest conftest for model-traced sweep validation.

Provides:
- Dynamic mesh device fixture (session-scoped, reopens on crash)
- tt-smi reset on test failure/timeout
- Vector loading from vectors_export/ as pytest parameters
- Result collection to results_export/
"""

import json
import os
import time

import pytest
import ttnn

import framework.tt_smi_util as tt_smi_util
from framework.sweeps_logger import sweeps_logger as logger


# ── Device fixture ──────────────────────────────────────────────────────────

class MeshDeviceManager:
    """Manages mesh device lifecycle with crash recovery."""

    def __init__(self):
        self.device = None
        self.arch_name = None
        self.mesh_shape = None
        self.reset_util = None

    def open(self, mesh_shape=None):
        if mesh_shape is None:
            mesh_env = os.environ.get("MESH_DEVICE_SHAPE", "").strip()
            if mesh_env and "x" in mesh_env:
                rows, cols = (int(x) for x in mesh_env.split("x"))
                mesh_shape = ttnn.MeshShape(rows, cols)
            else:
                mesh_shape = ttnn.MeshShape(1, 1)

        self.mesh_shape = mesh_shape
        self.device = ttnn.open_mesh_device(mesh_shape)
        self.arch_name = ttnn.get_arch_name()
        self.reset_util = tt_smi_util.ResetUtil(self.arch_name)
        logger.info(f"Opened mesh device: {mesh_shape}")
        return self.device

    def close(self):
        if self.device is not None:
            try:
                ttnn.close_mesh_device(self.device)
            except Exception as e:
                logger.warning(f"Error closing device: {e}")
            self.device = None

    def reopen(self):
        """Close and reopen the device after a crash/hang."""
        logger.warning("Reopening mesh device after failure...")
        self.close()
        if self.reset_util:
            try:
                self.reset_util.reset()
            except Exception as e:
                logger.error(f"tt-smi reset failed: {e}")
        time.sleep(1)
        return self.open(self.mesh_shape)


_device_manager = MeshDeviceManager()


@pytest.fixture(scope="session")
def mesh_device(request):
    """Session-scoped mesh device fixture."""
    device = _device_manager.open()
    yield (device, _device_manager.arch_name)
    _device_manager.close()


@pytest.fixture(autouse=True)
def _reopen_on_failure(request, mesh_device):
    """Reopen device if a test caused a device error."""
    yield
    if hasattr(request.node, "_device_error"):
        _device_manager.reopen()


# ── Hang/crash recovery hooks ──────────────────────────────────────────────

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        exc_info = call.excinfo
        if exc_info is not None:
            exc_str = str(exc_info.value)
            if "TIMEOUT" in exc_str or "unrecoverable" in exc_str or "hang" in exc_str.lower():
                item._device_error = True
                logger.warning(f"Device error detected in {item.nodeid}, will reopen device")


# ── Vector parametrization ─────────────────────────────────────────────────

def load_vectors_for_module(module_name, suite_name="model_traced"):
    """Load vectors from vectors_export/ for a given module."""
    vector_source = os.environ.get("TTNN_VECTORS_EXPORT_DIR", "tests/sweep_framework/vectors_export")
    vectors = []

    for fname in sorted(os.listdir(vector_source)):
        if not fname.startswith(module_name) or not fname.endswith(".json"):
            continue
        fpath = os.path.join(vector_source, fname)
        with open(fpath) as f:
            data = json.load(f)

        suite_data = data.get(suite_name, data)
        if not isinstance(suite_data, dict):
            continue

        for input_hash, vector in suite_data.items():
            if isinstance(vector, dict) and vector.get("validity") != "VectorValidity.INVALID":
                vector["input_hash"] = input_hash
                vectors.append(vector)

    return vectors


def vector_id(vector):
    """Generate a short test ID from a vector."""
    return vector.get("input_hash", "unknown")[:16]
