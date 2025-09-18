# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

# test cases provided in https://github.com/tenstorrent/tt-metal/issues/21069

TEST_CASES = [
    "tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_2d_multiple_output_blocks_per_core",
    "tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_in1_dram_sharded_tiny_tile",
    "tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_padded_1d_matmul",
]

NUM_REPEATS = 10
NUM_DEVICES_ENV_KEY = "USE_NUM_DEVICES"


@pytest.fixture
def use_num_devices_env():
    os.environ[NUM_DEVICES_ENV_KEY] = "1"
    yield
    os.environ.pop(NUM_DEVICES_ENV_KEY)


def test_matmul(use_num_devices_env):
    for _ in range(NUM_REPEATS):
        assert pytest.main(TEST_CASES) == pytest.ExitCode.OK
