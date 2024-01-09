# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
        (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384]), torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384]), torch.Size([1, 3, 320, 384])),
    ),
)
class TestEltwiseBinary:
    def test_ttnn_squared_difference(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.square(torch.sub(torch_a, torch_b))

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.squared_difference(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)

    @pytest.mark.parametrize("alpha", [1.0, 2.5, -2.5])
    def test_ttnn_subalpha(self, input_shape_a, input_shape_b, alpha, device):
        torch.manual_seed(0)

        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.sub(torch_a, torch_b, alpha=alpha)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.subalpha(a, b, alpha)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)
