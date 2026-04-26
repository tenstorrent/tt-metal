# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for YOLO26 tests.

Provides fixtures and command-line options for testing.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--input-size",
        action="store",
        default=640,
        type=int,
        choices=[320, 416, 512, 640, 1024],
        help="Input image size (default: 640)",
    )
    parser.addoption(
        "--variant",
        action="store",
        default="yolo26n",
        choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"],
        help="YOLO26 model variant (default: yolo26n)",
    )
    parser.addoption(
        "--batch-size",
        action="store",
        default=1,
        type=int,
        help="Batch size for inference (default: 1)",
    )


@pytest.fixture
def input_size(request):
    """Get input size from command line."""
    return request.config.getoption("--input-size")


@pytest.fixture
def variant(request):
    """Get model variant from command line."""
    return request.config.getoption("--variant")


@pytest.fixture
def batch_size(request):
    """Get batch size from command line."""
    return request.config.getoption("--batch-size")
