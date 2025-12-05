# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


def test_upsample(device):
    # Define input parameters
    batch_size, num_channels, height, width = 1, 64, 32, 32
    scale_h, scale_w = 2, 2

    # Create a random input tensor in NHWC format
    torch_input = torch.randn(batch_size, height, width, num_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )

    # Perform upsampling with scale factor
    scale_factor = (scale_h, scale_w)
    output = ttnn.upsample(tt_input, scale_factor)

    logger.info(f"Upsample output shape: {output.shape}")
    logger.info(f"Upsample output: {output}")
