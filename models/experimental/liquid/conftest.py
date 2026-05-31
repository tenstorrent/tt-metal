# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "liquid: LFM2.5-VL model tests")


@pytest.fixture(scope="session")
def liquid_weights_path():
    path = os.environ.get("LIQUID_WEIGHTS", os.path.expanduser("~/liquid_weights/"))
    if not os.path.exists(path):
        pytest.skip(f"LiquidAI weights not found at {path}. Set LIQUID_WEIGHTS env var.")
    return path


@pytest.fixture(scope="session")
def mesh_device_type():
    return os.environ.get("MESH_DEVICE", None)


@pytest.fixture(scope="session")
def is_multi_device(mesh_device_type):
    return mesh_device_type in ["N300", "T3K", "TG"]
