# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption(
        "--save-inputs",
        action="store_true",
        default=False,
        help="Generate permanent input folders under test_inputs/ and skip C++ execution.",
    )
