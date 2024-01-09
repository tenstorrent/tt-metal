# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_allclose


@pytest.mark.parametrize(
    "input_shape",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
class TestmseLossFucntion:
    def test_mse_loss_none(self, input_shape, device):
        torch_a = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape, dtype=torch.bfloat16)

        loss = torch.nn.MSELoss(reduction="none")

        torch_output_tensor = loss(torch_a.to(torch.float32), torch_b.to(torch.float32))

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.mseloss(a, b, ttnn.LOSS_MODE_NONE)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_mse_loss_sum(self, input_shape, device):
        torch_a = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape, dtype=torch.bfloat16)

        loss = torch.nn.MSELoss(reduction="sum")

        torch_output_tensor = loss(torch_a.to(torch.float32), torch_b.to(torch.float32))

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.mseloss(a, b, ttnn.LOSS_MODE_SUM)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_allclose(torch_output_tensor, output_tensor[0, 0, 0, 0], atol=4, rtol=1e-1)

    def test_mse_loss_mean(self, input_shape, device):
        if input_shape[3] == 384:
            pytest.skip("mean not supported in dimensions 3, 4")
        torch_a = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape, dtype=torch.bfloat16)

        loss = torch.nn.MSELoss(reduction="mean")

        torch_output_tensor = loss(torch_a.to(torch.float32), torch_b.to(torch.float32))

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.mseloss(a, b, ttnn.LOSS_MODE_MEAN)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_allclose(torch_output_tensor, output_tensor[0, 0, 0, 0], atol=4, rtol=1e-1)
