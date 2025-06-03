# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    """Add custom command line options to pytest"""
    parser.addoption(
        "--use_daemon",
        action="store_true",
        default=False,
        help="Use daemon mode for fabric EDM tests instead of direct execution",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ubench_quick_tests: quick tests for fast iteration")
    config.addinivalue_line("markers", "sanity_6u: quick subset of 6u applicable tests")
