# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def gtest_filter(request):
    return request.config.getoption("--gtest-filter")


@pytest.fixture
def no_profile(request):
    return request.config.getoption("--no-profile")


@pytest.fixture
def verbose_log(request):
    return request.config.getoption("--verbose-log")


@pytest.fixture
def plot(request):
    return request.config.getoption("--plot")


@pytest.fixture
def report(request):
    return request.config.getoption("--report")


@pytest.fixture
def arch(request):
    return request.config.getoption("--arch")


# These inputs override the default inputs used in data movement tests.
def pytest_addoption(parser):
    parser.addoption(
        "--no-profile",
        action="store_true",
        help="Use existing profiler logs. If not set, profile kernels and use those results.",
    )
    parser.addoption("--verbose-log", action="store_true", help="Enable verbose logging of profiling results.")
    parser.addoption(
        "--gtest-filter",
        action="store",
        default=None,
        help="Filter for gtest tests to run. If not set, all tests are run.",
    )
    parser.addoption("--plot", action="store_true", help="Export profiling plots to a .png file.")
    parser.addoption("--report", action="store_true", help="Export profiling results to a .csv file.")
    parser.addoption(
        "--arch",
        action="store",
        default=None,
        help="Architecture the tests are run on. If not set, defaults to the ARCH_NAME variable, or if that is not set, to 'blackhole'.",
    )
    parser.addoption(
        "--dm-help",
        action="store_true",
        help="Display a personalized help message for data movement tests.",
    )


# Handle the custom --dm-help option
def pytest_cmdline_main(config):
    if config.getoption("--dm-help"):
        print("\nData Movement Tests Help:")
        print("  --no-profile       Use existing profiler logs instead of profiling kernels.")
        print("  --verbose-log      Enable verbose logging of profiling results.")
        print("  --gtest-filter     Filter for gtest tests to run. If not set, all tests are run.")
        print("  --plot             Export profiling plots to a .png file.")
        print("  --report           Export profiling results to a .csv file.")
        print(
            "  --arch             Specify the architecture the tests are run on. If not set, defaults to first the ARCH_NAME variable, then to 'blackhole'."
        )
        print("\nExample Usage:")
        print("  pytest --no-profile --verbose-log --plot --report tests/tt_metal/tt_metal/data_movement")
        print("  pytest --gtest-filter='Directed' --arch='wormhole_b0' tests/tt_metal/tt_metal/data_movement")
        return 0
