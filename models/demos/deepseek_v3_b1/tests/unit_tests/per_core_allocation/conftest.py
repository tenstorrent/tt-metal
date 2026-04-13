# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest that enables HYBRID allocator mode for all tests in this directory.
The env var is set before device creation and removed on teardown.
"""

import os

import pytest


@pytest.fixture(autouse=True, scope="module")
def enable_hybrid_allocator():
    """Enable HYBRID allocator mode for all tests in this module via env var."""
    os.environ["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"
    yield
    os.environ.pop("TT_METAL_ALLOCATOR_MODE_HYBRID", None)
