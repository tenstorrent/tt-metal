# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


def test_embedding(device):
    # device_id = 0
    # device = ttnn.open_device(device_id=device_id)

    # Create a tensor containing indices into the embedding matrix
    tensor = ttnn.to_device(
        ttnn.from_torch(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]), dtype=ttnn.uint32), device=device
    )
    # Create an embedding matrix containing 10 tensors of size 4
    weight = ttnn.rand((10, 4), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Perform embedding lookup
    output = ttnn.embedding(tensor, weight)
    logger.info(f"Output: {output}")
