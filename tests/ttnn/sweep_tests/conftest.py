# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_make_parametrize_id(config, val, argname):
    return f"{argname}={val}"


def pytest_addoption(parser):
    parser.addoption(
        "--run-sweep-at-index", action="store", default=None, type=int, help="Index of the sweep to reproduce."
    )


@pytest.fixture(name="sweep_index", scope="session", autouse=True)
def sweep_index(request):
    yield request.config.getoption("--run-sweep-at-index")
