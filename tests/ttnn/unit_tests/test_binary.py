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
    def test_squared_difference(self, input_shape_a, input_shape_b, device):
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

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    @pytest.mark.parametrize("alpha", [1.0, 2.5, -2.5])
    def test_subalpha(self, input_shape_a, input_shape_b, alpha, device):
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

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    @pytest.mark.parametrize("alpha", [1.0, 2.5, -2.5])
    def test_addalpha(self, input_shape_a, input_shape_b, alpha, device):
        torch.manual_seed(0)

        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.add(torch_a, torch_b, alpha=alpha)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.addalpha(a, b, alpha)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_atan2(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.atan2(torch_b, torch_a)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.atan2(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_hypot(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.hypot(torch_a, torch_b)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.hypot(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_ldexp(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.ldexp(torch_a, torch_b)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.ldexp(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_logaddexp(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.logaddexp(torch_a, torch_b)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.logaddexp(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_logaddexp2(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.logaddexp2(torch_a, torch_b)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.logaddexp2(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.98)

    def test_xlogy(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.xlogy(torch_a, torch_b)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.xlogy(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_logical_or(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.logical_or(torch_a, torch_b)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.logical_or(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_nextafter(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.nextafter(torch_a, torch_b)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.nextafter(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_logical_xor(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.logical_xor(torch_a, torch_b)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.logical_xor(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_logical_and(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.logical_and(torch_a, torch_b)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.logical_and(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

    def test_assign(self, input_shape_a, input_shape_b, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_b.copy_(torch_a)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.assign(a, b)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_b, output_tensor, 0.99)

    @pytest.mark.parametrize(
        "rtol, atol, equal_nan",
        (
            (1e-3, 1e-2, False),
            (1e-5, 1e-4, True),
            (1e-7, 1e-6, True),
        ),
    )
    def test_isclose(self, input_shape_a, input_shape_b, rtol, atol, equal_nan, device):
        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        torch_output_tensor = torch.isclose(torch_a, torch_b, rtol=rtol, atol=atol, equal_nan=equal_nan)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = ttnn.isclose(a, b, rtol, atol, equal_nan)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
