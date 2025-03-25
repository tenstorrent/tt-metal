# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull

# def run_sort_test(N, C, H, W, dim, descending, device):
#     torch.manual_seed(2005)
#     shape = [N, C, H, W]
#     torch_dtype = torch.bfloat16
#     input = torch.randn(shape, dtype=torch_dtype) * 0.9
#     pyt_sort_values, pyt_sort_indices = torch.sort(input, dim=dim, descending=descending)
#     ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
#     ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)
#     assert list(ttnn_sort_values.shape) == shape
#     assert list(ttnn_sort_indices.shape) == shape
#     ttnn_torch_values = ttnn.to_torch(ttnn_sort_values)
#     ttnn_torch_indices = ttnn.to_torch(ttnn_sort_indices).to(torch.int64)
#     assert_with_pcc(pyt_sort_values, ttnn_torch_values)
#     assert_with_pcc(pyt_sort_indices, ttnn_torch_indices)

# def test_output_shape(device):
#     run_sort_test(1, 1, 1, 1, 0, False, device)
#     run_sort_test(1, 1, 1, 1, 1, False, device)
#     run_sort_test(1, 1, 1, 1, 2, False, device)
#     run_sort_test(1, 1, 1, 1, 3, False, device)
#     run_sort_test(1, 1, 1, 1, 0, True, device)
#     run_sort_test(1, 1, 1, 1, 1, True, device)
#     run_sort_test(1, 1, 1, 1, 2, True, device)
#     run_sort_test(1, 1, 1, 1, 3, True, device)


def test_sort_output_shape(device):
    torch.manual_seed(2005)
    shape = [3, 3]
    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)
    print(input)
    ttnn.topk(input, 1, dim=0, largest=False, sorted=True)
    # pyt_sort_values, pyt_sort_indices = torch.sort(input, dim=0, descending=False)
    # ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    # ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=0, descending=False)
    # assert list(ttnn_sort_values.shape) == shape
    # assert list(ttnn_sort_indices.shape) == shape
    # ttnn_torch_values = ttnn.to_torch(ttnn_sort_values)
    # ttnn_torch_indices = ttnn.to_torch(ttnn_sort_indices).to(torch.int64)
    # assert_with_pcc(pyt_sort_values, ttnn_torch_values)
    # assert_with_pcc(pyt_sort_indices, ttnn_torch_indices)
