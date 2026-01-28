# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    """Add custom command line options for pytest"""
    parser.addoption(
        "--test-modules",
        action="store",
        default="all",
        help="Comma-separated list of modules to test. Options: all, attention, rms_norm, router, experts, mlp, decoder. Example: --test-modules=attention,mlp",
    )


@pytest.fixture
def test_modules(request):
    """Fixture to get the test_modules value from command line or use default 'all'"""
    return request.config.getoption("--test-modules")
