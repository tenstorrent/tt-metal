# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest
import torch
import ttnn

# Make the operation module importable without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "operation"))
from mul_relu_add_overlap_op import MulReluAddOverlapConfig, mul_relu_add_overlap_op  # noqa: E402

from tests.ttnn.utils_for_testing import assert_with_ulp


SHAPES = [
    pytest.param([32, 32], id="1tile"),
    pytest.param([32, 128], id="1x4tiles"),
    pytest.param([2, 4, 32, 64], id="4d"),
    pytest.param([16, 16, 1024, 1024], id="4d_large"),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_mul_relu_add_overlap(device, shape):
    torch.manual_seed(0)

    a_torch = torch.full(shape, 2.0, dtype=torch.bfloat16)
    # b, c values in {-1, 0, 1}: integer uniform in [-1, 1].
    b_torch = torch.randint(-1, 2, shape).to(torch.bfloat16)
    c_torch = torch.randint(-1, 2, shape).to(torch.bfloat16)

    golden = torch.relu(a_torch * b_torch) + c_torch

    a = ttnn.from_torch(
        a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        b_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    c = ttnn.from_torch(
        c_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = mul_relu_add_overlap_op(a, b, c, MulReluAddOverlapConfig())
    result_torch = ttnn.to_torch(result)

    print(f"golden = \n{golden}")
    print(f"output = \n{result_torch}")

    assert_with_ulp(golden, result_torch, ulp_threshold=2)
