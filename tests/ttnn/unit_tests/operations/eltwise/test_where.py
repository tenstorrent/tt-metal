# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


def test_ttnn_where(device):
    C = torch.ones(4, 4, dtype=torch.float32)
    T = torch.randn(4, 4, dtype=torch.float32)
    F = torch.ones(4, 4, dtype=torch.float32) * 10
    golden = torch.where(C != 0, T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)
    print(result)
    print(golden)


def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))


def test_ttnn_where_nan(device):
    dtype = torch.float32

    condition = torch.tensor([1, 0, -2, 0, 5, 0, 0, 8, 0, -1], dtype=dtype)
    condition_all_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dtype)
    condition_all_zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype)

    # true and false value tensors
    true_values = torch.tensor(
        [1.0, float("nan"), 3.0, float("inf"), -float("inf"), -1.0, 0.0, -0.0, 42.49, -92.42], dtype=dtype
    )
    false_values = torch.tensor(
        [-1.0, 999.9, float("nan"), -float("inf"), float("inf"), 1.0, -0.0, 0.0, -3.14, 7.84], dtype=dtype
    )

    ttnn_condition = ttnn.from_torch(condition, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_condition_all_ones = ttnn.from_torch(
        condition_all_ones, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_condition_all_zeros = ttnn.from_torch(
        condition_all_zeros, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_true_values = ttnn.from_torch(true_values, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_false_values = ttnn.from_torch(false_values, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_result1 = ttnn.where(ttnn_condition, ttnn_true_values, ttnn_false_values)
    ttnn_result2 = ttnn.where(ttnn_condition_all_ones, ttnn_true_values, ttnn_false_values)
    ttnn_result3 = ttnn.where(ttnn_condition_all_zeros, ttnn_true_values, ttnn_false_values)

    tt_result1 = ttnn.to_torch(ttnn_result1)
    tt_result2 = ttnn.to_torch(ttnn_result2)
    tt_result3 = ttnn.to_torch(ttnn_result3)

    print("ttnn res", tt_result1)
    print("ttnn res", tt_result2, torch_equal_nan(tt_result2, true_values))
    print("ttnn res", tt_result3, torch_equal_nan(tt_result3, false_values))

    # where operation in torch expects condition to be a boolean dtype, in ttnn.where we follow 0's & non-zero's (0's and 1's would be ideal)
    result1 = torch.where(condition.bool(), true_values, false_values)
    result2 = torch.where(condition_all_ones.bool(), true_values, false_values)
    result3 = torch.where(condition_all_zeros.bool(), true_values, false_values)
    print(result1)
    print(result2, torch_equal_nan(result2, true_values))
    print(result3, torch_equal_nan(result3, false_values))

    assert torch_equal_nan(tt_result1, result1)
    assert torch_equal_nan(tt_result2, result2)
    assert torch_equal_nan(tt_result3, result3)
