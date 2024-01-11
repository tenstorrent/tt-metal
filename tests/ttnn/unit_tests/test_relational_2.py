# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc


def from_device(tensor: ttnn.Tensor):
    tensor = ttnn.from_device(tensor)
    tensor = ttnn.to_torch(tensor)
    return tensor


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
class TestEltwiseRelational:
    def test_ttnn_gt(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.gt(a, b)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a > torch_b).to(torch.bfloat16), from_device(output))
        ttnn.close(device)

    def test_ttnn_gte(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.gte(a, b)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a >= torch_b).to(torch.bfloat16), from_device(output))

        ttnn.close(device)

    def test_ttnn_eq(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.eq(a, b)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a == torch_b).to(torch.bfloat16), from_device(output))
        ttnn.close(device)

    def test_ttnn_lt(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.lt(a, b)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a < torch_b).to(torch.bfloat16), from_device(output))
        ttnn.close(device)

    def test_ttnn_lte(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.lte(a, b)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a <= torch_b).to(torch.bfloat16), from_device(output))
        ttnn.close(device)

    def test_ttnn_ne(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.ne(a, b)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a != torch_b).to(torch.bfloat16), from_device(output))
        ttnn.close(device)

    def test_ttnn_gtz(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)

        a = ttnn.to_device(a, device)

        output = ttnn.gtz(a)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a > 0).to(torch.bfloat16), from_device(output))
        ttnn.close(device)

    def test_ttnn_gez(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)

        a = ttnn.to_device(a, device)

        output = ttnn.gez(a)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a >= 0).to(torch.bfloat16), from_device(output))
        ttnn.close(device)

    def test_ttnn_eqz(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)

        a = ttnn.to_device(a, device)

        output = ttnn.eqz(a)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a == 0).to(torch.bfloat16), from_device(output))
        ttnn.close(device)

    def test_ttnn_ltz(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)

        a = ttnn.to_device(a, device)

        output = ttnn.ltz(a)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a < 0).to(torch.bfloat16), from_device(output))
        ttnn.close(device)

    def test_ttnn_nez(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)

        a = ttnn.to_device(a, device)

        output = ttnn.nez(a)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")
        # print(f"first row: {output[:1]}")

        assert_with_pcc((torch_a != 0).to(torch.bfloat16), from_device(output))
        ttnn.close(device)
