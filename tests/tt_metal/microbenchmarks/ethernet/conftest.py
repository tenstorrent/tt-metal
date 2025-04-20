# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "ubench_quick_tests: quick tests for fast iteration")
