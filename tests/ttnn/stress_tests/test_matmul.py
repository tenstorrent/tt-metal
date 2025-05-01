# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

# test cases provided in https://github.com/tenstorrent/tt-metal/issues/21069

TEST_CASES = [
    "tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_2d_multiple_output_blocks_per_core",
    "tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_in1_dram_sharded_tiny_tile",
    "tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_padded_1d_matmul",
]

NUM_REPEATS = 5


def test_matmul():
    for _ in range(NUM_REPEATS):
        pytest.main(TEST_CASES)
