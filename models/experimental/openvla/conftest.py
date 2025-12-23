# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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
