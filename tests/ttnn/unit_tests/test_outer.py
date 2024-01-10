# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("s", [32, 64, 128])
def test_outer(device, s):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((1, 1, s * 2, 1), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((1, 1, 1, s), dtype=torch.bfloat16)

    torch_output_tensor = torch.outer(torch_input_tensor_a.squeeze(), torch_input_tensor_b.squeeze())

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)

    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)

    output_tensor = ttnn.outer(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor[0, 0], 0.9998)
