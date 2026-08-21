# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

from tests.ttnn.unit_tests.operations.examples.report_gate import ENABLE_VAR


def pytest_addoption(parser):
    parser.addoption(
        "--write-reports",
        action="store_true",
        default=False,
        help="Refresh the checked-in report.md of each example. Off by default: the numbers are "
        "box- and arch-specific, so a normal run would leave a spurious diff.",
    )


def pytest_configure(config):
    # Registered only when this directory is one of the invocation targets; when the whole test
    # tree is collected from further up, the flag does not exist and the env var is the way in.
    if config.getoption("--write-reports", default=False):
        os.environ[ENABLE_VAR] = "1"
