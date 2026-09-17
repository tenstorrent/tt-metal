# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


def test_quantize(device):
    # Create a float tensor to quantize
    input_tensor = ttnn.from_torch(
        torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Quantize the input onto the int8 range, giving [[-127, -42], [43, 127]]
    scale = 0.001173
    zero_point = -213
    output = ttnn.quantize(input_tensor, scale, zero_point)
    logger.info(f"Quantize result: {output}")


def test_requantize(device):
    # Create a quantized tensor to requantize
    input_tensor = ttnn.from_torch(
        torch.tensor([[-127, -42], [43, 127]], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Requantize onto a different scale and zero point, [-127, 127] -> [-36, 73]
    in_scale = 0.001173
    in_zero_point = -213
    out_scale = 0.002727
    out_zero_point = -73
    output = ttnn.requantize(input_tensor, in_scale, in_zero_point, out_scale, out_zero_point)
    logger.info(f"Requantize result: {output}")


def test_dequantize(device):
    # Create a quantized tensor to dequantize
    input_tensor = ttnn.from_torch(
        torch.tensor([[-127, -42], [43, 127]], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Dequantize back to float, [-127, 127] -> approximately [0.1, 0.4]
    scale = 0.001173
    zero_point = -213
    output = ttnn.dequantize(input_tensor, scale, zero_point)
    logger.info(f"Dequantize result: {output}")
