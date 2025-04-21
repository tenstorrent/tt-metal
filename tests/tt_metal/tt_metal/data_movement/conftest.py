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
    return request.config.getoption("--verbose_log")


@pytest.fixture
def plot(request):
    return request.config.getoption("--plot")


@pytest.fixture
def report(request):
    return request.config.getoption("--report")


# These inputs override the default inputs used in data movement tests.
def pytest_addoption(parser):
    parser.addoption(
        "--no-profile",
        action="store_true",
        help="Use existing profiler logs. If not set, profile kernels and use those results.",
    )
    parser.addoption("--verbose_log", action="store_true", help="Enable verbose logging of profiling results.")
    parser.addoption(
        "--gtest-filter",
        action="store",
        default=None,
        help="Filter for gtest tests to run. If not set and profile flag is set, all tests are run.",
    )
    parser.addoption("--plot", action="store_true", help="Export profiling plots to a .png file.")
    parser.addoption("--report", action="store_true", help="Export profiling results to a .csv file.")
