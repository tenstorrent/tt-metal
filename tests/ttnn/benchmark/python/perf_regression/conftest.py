# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for performance regression tests.
"""

import os
from pathlib import Path
import pytest
import ttnn


def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption(
        "--perf-baseline-path",
        action="store",
        default=None,
        dest="perf_baseline_path",
        help="Path to baseline directory (default: uses TT_METAL_HOME/tests/ttnn/benchmark/python/perf_regression/gt/accessor_benchmarks/<arch>)",
    )


@pytest.fixture
def perf_baseline_path(request):
    """Fixture to get baseline path from command line or use default."""
    custom_path = request.config.getoption("perf_baseline_path")
    if custom_path:
        return Path(custom_path)

    # Default path
    tt_metal_home = os.environ.get("TT_METAL_HOME")
    if not tt_metal_home:
        raise RuntimeError("TT_METAL_HOME environment variable not set")

    return (
        Path(tt_metal_home)
        / "tests"
        / "ttnn"
        / "benchmark"
        / "python"
        / "perf_regression"
        / "gt"
        / "accessor_benchmarks"
        / ttnn.get_arch_name()
    )
