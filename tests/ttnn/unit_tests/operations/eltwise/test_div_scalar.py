# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [64])
def test_div_1D_tensor_and_scalar(device, scalar, size):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor / scalar

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.divide(input_tensor, scalar)
    output_tensor = ttnn.to_torch(output_tensor, torch_rank=1)

    print("torch_output_tensor", torch_output_tensor)
    print("output_tensor", output_tensor)
    print(ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor))

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.9999
    assert output_tensor.shape == (size,)
