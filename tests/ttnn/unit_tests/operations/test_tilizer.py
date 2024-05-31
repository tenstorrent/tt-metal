# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger


def test_device_tilize(device):
    """Benchmark host vs. device tilizer for converting torch tensor to tilized tensor."""
    import time

    torch_tensor = torch.randn((4544, 18176), dtype=torch.bfloat16)
    output_dtype = ttnn.bfloat8_b

    start = time.time()
    tensor = ttnn.from_torch(torch_tensor, dtype=output_dtype, layout=ttnn.TILE_LAYOUT)
    end = time.time()
    logger.info(f"Time taken to convert to tensor using host-tilizer: {end-start}")

    start = time.time()
    tensor = ttnn.from_torch(
        torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT, dtype=output_dtype, device=device)
    ttnn.synchronize_device(device)
    end = time.time()
    logger.info(f"Time taken to convert to tensor using device-tilizer: {end-start}")
