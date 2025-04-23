# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger


def pytest_configure(config):
    config.addinivalue_line("markers", "ubench_quick_tests: quick tests for fast iteration")
    config.addinivalue_line("markers", "sanity_6u: quick subset of 6u applicable tests")


def pytest_addoption(parser):
    parser.addoption(
        "--arch",
        action="store",
        default="wh",
        help="architecture name. Valid options are 'wh' for Wormhole or 'bh' for Blackhole",
    )
    parser.addoption(
        "--machine-type", action="store", default="t3k", help="machine type suffix. Valid options are 't3k' or '6u'"
    )


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    arch_name_arg = metafunc.config.option.arch
    arch_name = arch_name_arg

    machine_type_arg = metafunc.config.option.machine_type

    assert machine_type_arg is not None
    machine_type = machine_type_arg

    # The arch_name and machine_type fixtures are *required* for all bandwidth tests. We check to ensure they are defined.
    if "arch_name" not in metafunc.fixturenames:
        raise Exception(f"arch_name fixture is not defined in test {metafunc.function.__name__}")
    metafunc.parametrize("arch_name", [arch_name], scope="session")

    if "machine_type" not in metafunc.fixturenames:
        raise Exception(f"machine_type fixture is not defined in test {metafunc.function.__name__}")
    metafunc.parametrize("machine_type", [machine_type], scope="session")
