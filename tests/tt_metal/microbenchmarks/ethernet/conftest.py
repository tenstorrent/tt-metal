# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    """Add custom command line options to pytest"""
    parser.addoption(
        "--direct_exec",
        action="store_true",
        default=False,
        help="Use direct execution instead of daemon mode for fabric EDM tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ubench_quick_tests: quick tests for fast iteration")
    config.addinivalue_line("markers", "sanity_6u: quick subset of 6u applicable tests")
    config.addinivalue_line("markers", "extended: extended/exhaustive sweep tests (require --extended flag)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--extended"):
        skip_extended = pytest.mark.skip(reason="extended sweep test — pass --extended to run")
        for item in items:
            if "extended" in item.keywords:
                item.add_marker(skip_extended)
