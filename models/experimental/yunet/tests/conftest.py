# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for YUNet tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--input-size",
        action="store",
        default="640",
        choices=["320", "640"],
        help="Input size for YUNet model (320 or 640). Default: 640",
    )


@pytest.fixture
def input_size(request):
    """Fixture to get the input size from command line."""
    size = int(request.config.getoption("--input-size"))
    return size, size  # Returns (height, width)
