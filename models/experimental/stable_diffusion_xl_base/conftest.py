# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_CI_WEIGHTS_PATH


def pytest_addoption(parser):
    parser.addoption(
        "--start-from",
        action="store",
        default=0,
        help="Start from prompt number (0-4999)",
    )
    parser.addoption(
        "--num-prompts",
        action="store",
        default=5000,
        help="Number of prompts to process (default: 5000)",
    )
    parser.addoption(
        "--loop-iter-num",
        action="store",
        default=10,
        help="Number of iterations of denoising loop (default: 10)",
    )


@pytest.fixture
def evaluation_range(request):
    start_from = request.config.getoption("--start-from")
    num_prompts = request.config.getoption("--num-prompts")
    if start_from is not None:
        start_from = int(start_from)
    else:
        start_from = 0

    if num_prompts is not None:
        num_prompts = int(num_prompts)
    else:
        num_prompts = 5000

    return start_from, num_prompts


@pytest.fixture
def loop_iter_num(request):
    return int(request.config.getoption("--loop-iter-num"))


@pytest.fixture(scope="session", autouse=True)
def set_weights_path():
    if os.getenv("CI") == "true":
        os.environ["HF_HOME"] = SDXL_CI_WEIGHTS_PATH
