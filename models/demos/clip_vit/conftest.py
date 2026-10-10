# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption(
        "--opt",
        action="store_true",
        default=False,
        help="Use TtCLIPModelOptimized instead of the baseline TtCLIPModel.",
    )
    parser.addoption(
        "--b8",
        action="store_true",
        default=False,
        help="Run the optimized model with bfloat8_b dtype (requires --opt).",
    )
    parser.addoption(
        "--b16",
        action="store_true",
        default=False,
        help="Run the optimized model with bfloat16 dtype (requires --opt). Default when --opt is set.",
    )
