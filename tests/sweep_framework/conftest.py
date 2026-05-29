# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

def pytest_addoption(parser):
    parser.addoption(
        "--module-name",
        action="store",
        default=None,
        help="Comma-separated list of sweep module names to run",
    )
    parser.addoption(
        "--suite-name",
        action="store",
        default="model_traced",
        help="Suite name filter (default: model_traced)",
    )
    parser.addoption(
        "--vector-source",
        action="store",
        default=None,
        help="Path to vectors_export/ directory",
    )
