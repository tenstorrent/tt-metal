# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for YUNet tests."""

import pytest

from models.experimental.yunet.common import setup_yunet_reference


@pytest.fixture(scope="session", autouse=True)
def yunet_reference():
    """Clone the upstream YUNet reference repo before any test in this tree runs.

    The reference nets and weights/best.pt are not vendored. Every import of
    models.experimental.yunet.YUNet is lazy -- common.load_torch_model and
    test_pcc both do it inside the function body -- so setting this up at test
    setup time is early enough; nothing resolves it at collection. Session-scoped
    and idempotent: setup_yunet_reference() no-ops once the directory exists.
    """
    setup_yunet_reference()


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
