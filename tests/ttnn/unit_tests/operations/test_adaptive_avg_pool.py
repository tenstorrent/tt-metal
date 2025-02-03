# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn


shapes_and_output_sizes = [
    ((1, 3, 8, 8), (4, 4)),
    ((1, 3, 8, 16), (4, 8)),
    ((1, 1, 10, 10), (5, 5)),
    ((1, 3, 8, 8), (2, 2)),
    ((1, 6, 64, 64), (16, 16)),
    ((1, 3, 3, 3), (1, 1)),
    ((1, 3, 7, 13), (3, 5)),
    ((1, 4, 8, 16), (8, 8)),
]


@pytest.mark.parametrize("input_shape, output_size", shapes_and_output_sizes)
def test_run_adaptive_avg_pool2d(device, input_shape, output_size):
    dtype = ttnn.float32

    torch.manual_seed(0)
    torch_input_tensor = torch.randn(input_shape)
    torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor, output_size)
    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, device=device)
    output_tensor = ttnn.adaptive_avg_pool2d(input_tensor, ttnn.Shape(output_size))
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    assert_with_pcc(torch_output_tensor, output_tensor)


shapes_and_output_sizes_1d = [
    ((1, 3, 8), (4,)),
    ((1, 1, 10), (5,)),
    ((2, 4, 16), (8,)),
    ((3, 2, 12), (6,)),
    ((1, 5, 20), (10,)),
    ((4, 3, 15), (3,)),
]


@pytest.mark.parametrize("input_shape, output_size", shapes_and_output_sizes_1d)
def test_run_adaptive_avg_pool1d(device, input_shape, output_size):
    dtype = ttnn.float32

    torch.manual_seed(0)
    torch_input_tensor = torch.randn(input_shape)

    torch_output_tensor = torch.nn.functional.adaptive_avg_pool1d(torch_input_tensor, output_size[0])
    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 1))
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, device=device)
    output_tensor = ttnn.adaptive_avg_pool1d(input_tensor, ttnn.Shape(output_size))

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 2, 1))

    assert_with_pcc(torch_output_tensor, output_tensor)
