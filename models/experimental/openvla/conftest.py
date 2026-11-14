# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
OpenVLA pytest configuration and fixtures.

Usage:
    # Single device (P150/N150)
    OPENVLA_WEIGHTS=~/openvla_weights/ pytest models/experimental/openvla/tests/ -v

    # Multi-device (N300)
    MESH_DEVICE=N300 OPENVLA_WEIGHTS=~/openvla_weights/ pytest models/experimental/openvla/tests/ -v
"""

import os

import pytest


def pytest_configure(config):
    """Configure pytest for OpenVLA tests."""
    config.addinivalue_line("markers", "openvla: OpenVLA model tests")


@pytest.fixture(scope="session")
def openvla_weights_path():
    """Get OpenVLA weights path from environment."""
    path = os.environ.get("OPENVLA_WEIGHTS", os.path.expanduser("~/openvla_weights/"))
    if not os.path.exists(path):
        pytest.skip(f"OpenVLA weights not found at {path}. Set OPENVLA_WEIGHTS env var.")
    return path


@pytest.fixture(scope="session")
def mesh_device_type():
    """Get mesh device type from environment."""
    return os.environ.get("MESH_DEVICE", None)


@pytest.fixture(scope="session")
def is_multi_device(mesh_device_type):
    """Check if running on multi-device setup."""
    return mesh_device_type in ["N300", "T3K", "TG"]


@pytest.fixture
def device_params(request):
    """Fixture ensuring single device (N150/P150) does not request fabric_config."""
    params = getattr(request, "param", {}).copy()
    callspec = getattr(request.node, "callspec", None)
    mesh_device = callspec.params.get("mesh_device") if callspec else None
    if mesh_device is None:
        mesh_device = os.environ.get("MESH_DEVICE", "N150")

    is_single_device = False
    if isinstance(mesh_device, tuple):
        is_single_device = (mesh_device == (1, 1))
    elif isinstance(mesh_device, int):
        is_single_device = (mesh_device == 1)
    elif isinstance(mesh_device, str):
        is_single_device = mesh_device in ["N150", "P150"]

    if is_single_device and "fabric_config" in params:
        params["fabric_config"] = None

    return params
