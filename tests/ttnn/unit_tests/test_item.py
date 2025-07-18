# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import ttnn
import pytest


@pytest.mark.parametrize(
    "torch_tensor, ttnn_dtype, description",
    [
        (torch.tensor([100], dtype=torch.uint8), ttnn.uint8, "UINT8"),
        (torch.tensor([-100], dtype=torch.int32), ttnn.int32, "INT32"),
        (torch.tensor([3.14], dtype=torch.float32), ttnn.float32, "FLOAT32"),
        (torch.tensor([2.71], dtype=torch.bfloat16), ttnn.bfloat16, "BFLOAT16"),
        (torch.tensor([30000], dtype=torch.int16), ttnn.uint16, "UINT16"),
        (torch.tensor([4294967295], dtype=torch.int64), ttnn.uint32, "UINT32"),
        (torch.tensor([1.5], dtype=torch.float32), ttnn.bfloat8_b, "BFLOAT8_B"),
        (torch.tensor([2.5], dtype=torch.float32), ttnn.bfloat4_b, "BFLOAT4_B"),
        (torch.tensor([200], dtype=torch.uint8), ttnn.uint8, "UINT8 edge case"),
        # Zero values
        (torch.tensor([0], dtype=torch.uint8), ttnn.uint8, "UINT8 zero"),
        (torch.tensor([0], dtype=torch.int32), ttnn.int32, "INT32 zero"),
        (torch.tensor([0.0], dtype=torch.float32), ttnn.float32, "FLOAT32 zero"),
        (torch.tensor([0], dtype=torch.int64), ttnn.uint32, "UINT32 zero"),
        (torch.tensor([0.0], dtype=torch.float32), ttnn.bfloat8_b, "BFLOAT8_B zero"),
        (torch.tensor([0.0], dtype=torch.float32), ttnn.bfloat4_b, "BFLOAT4_B zero"),
        # Maximum values
        (torch.tensor([255], dtype=torch.uint8), ttnn.uint8, "UINT8 max"),
        (torch.tensor([2147483647], dtype=torch.int32), ttnn.int32, "INT32 max"),
        (torch.tensor([4294967295], dtype=torch.int64), ttnn.uint32, "UINT32 max"),
        # Minimum values
        (torch.tensor([-2147483648], dtype=torch.int32), ttnn.int32, "INT32 min"),
        # Small float values
        (torch.tensor([1e-6], dtype=torch.float32), ttnn.float32, "FLOAT32 small"),
        (torch.tensor([-1e-6], dtype=torch.float32), ttnn.float32, "FLOAT32 negative small"),
        # Infinity values
        (torch.tensor([float("inf")], dtype=torch.float32), ttnn.float32, "FLOAT32 +inf"),
        (torch.tensor([float("-inf")], dtype=torch.float32), ttnn.float32, "FLOAT32 -inf"),
        # NaN values
        (torch.tensor([float("nan")], dtype=torch.float32), ttnn.float32, "FLOAT32 nan"),
        # Very large and small numbers
        (torch.tensor([1e20], dtype=torch.float32), ttnn.float32, "FLOAT32 large"),
        (torch.tensor([-1e20], dtype=torch.float32), ttnn.float32, "FLOAT32 large negative"),
        (torch.tensor([1e-20], dtype=torch.float32), ttnn.float32, "FLOAT32 very small"),
        # BFLOAT16 precision test
        (torch.tensor([0.123456789], dtype=torch.bfloat16), ttnn.bfloat16, "BFLOAT16 precision"),
        # 2D
        (torch.tensor([[0.123456789]], dtype=torch.bfloat16), ttnn.bfloat16, "2D BFLOAT16 precision"),
        (torch.tensor([[2147483647]], dtype=torch.int32), ttnn.int32, "2D INT32 max"),
        (torch.tensor([[float("-inf")]], dtype=torch.float32), ttnn.float32, "2D FLOAT32 -inf"),
        # 3D
        (torch.tensor([[[30000]]], dtype=torch.int16), ttnn.uint16, "3D UINT16"),
        (torch.tensor([[[0.0]]], dtype=torch.float32), ttnn.float32, "3D FLOAT32 zero"),
        (torch.tensor([[[float("nan")]]], dtype=torch.float32), ttnn.float32, "3D FLOAT32 nan"),
        # 4D
        (torch.tensor([[[[30000]]]], dtype=torch.int16), ttnn.uint16, "4D UINT16"),
        (torch.tensor([[[[0.0]]]], dtype=torch.float32), ttnn.float32, "4D FLOAT32 zero"),
        (torch.tensor([[[[float("nan")]]]], dtype=torch.float32), ttnn.float32, "4D FLOAT32 nan"),
        (torch.tensor([[[[1000000]]]], dtype=torch.int64), ttnn.uint32, "4D UINT32"),
        (torch.tensor([[[[3.14]]]], dtype=torch.float32), ttnn.bfloat8_b, "4D BFLOAT8_B"),
        (torch.tensor([[[[2.71]]]], dtype=torch.float32), ttnn.bfloat4_b, "4D BFLOAT4_B"),
    ],
)
def test_tensor_item_basic_types(device, torch_tensor, ttnn_dtype, description):
    """Test .item() with basic data types"""

    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    if ttnn_dtype == ttnn.bfloat8_b or ttnn_dtype == ttnn.bfloat4_b:
        torch_tensor = ttnn.to_torch(ttnn_tensor)  # Torch does not support bfloat8_b and bfloat4_b

    original_value = torch_tensor.item()
    ttnn_value = ttnn_tensor.item()

    if math.isnan(original_value):
        assert math.isnan(ttnn_value), f"{description}: {original_value} != {ttnn_value}"
    elif math.isinf(original_value):
        assert math.isinf(ttnn_value), f"{description}: {original_value} != {ttnn_value}"
    else:
        assert original_value == ttnn_value, f"{description}: {original_value} != {ttnn_value}"
