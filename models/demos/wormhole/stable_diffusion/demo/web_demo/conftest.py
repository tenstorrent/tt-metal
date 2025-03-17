# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest


# This will add a --port argument to pytest CLI
def pytest_addoption(parser):
    parser.addoption("--port", action="store", default=7000, help="Port to run the server")


# Fixture to get the value of --port passed via pytest CLI
@pytest.fixture
def port(request):
    return request.config.getoption("--port")
