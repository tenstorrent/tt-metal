# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--loop-iter-num",
        action="store",
        default=20,
        help="Number of iterations of denoising loop (default: 20)",
    )


@pytest.fixture
def loop_iter_num(request):
    return int(request.config.getoption("--loop-iter-num"))
