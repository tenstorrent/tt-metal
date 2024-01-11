# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from types import ModuleType

import pytest

import ttnn


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, ModuleType):
        val = val.__name__
    return f"{argname}={val}"


def pytest_addoption(parser):
    parser.addoption(
        "--ttnn-enable-debug-decorator", action="store_const", const=True, help="Enable ttnn debug decorator"
    )


@pytest.fixture(name="ttnn_enable_debug_decorator_from_cli", scope="session", autouse=True)
def ttnn_enable_debug_decorator_from_cli(request):
    if request.config.getoption("--ttnn-enable-debug-decorator"):
        ttnn.decorators.ENABLE_DEBUG_DECORATOR = True
        yield
        ttnn.decorators.ENABLE_DEBUG_DECORATOR = False
    else:
        yield


@pytest.fixture
def ttnn_enable_debug_decorator():
    ttnn.decorators.ENABLE_DEBUG_DECORATOR = True
    yield
    ttnn.decorators.ENABLE_DEBUG_DECORATOR = False
