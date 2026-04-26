# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.demos.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_tensor(device):
    # Create tensor with exact memory config from failing conv2d input
    B, H, W, C = 1, 128, 128, 960
    data = torch.randn(B, 1, H * W, C)

    # L1 block sharded config matching down_blocks.1.resnets.0.norm1
    mem_cfg = ttnn.create_sharded_memory_config(
        shape=(B, 1, H * W, C),
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    tensor = ttnn.from_torch(
        data, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem_cfg
    )
    print(f"Shape: {tensor.shape}, Layout: {tensor.layout}, Memory: {tensor.memory_config()}")

    # Reshard to (x=10, y=8) core grid
    new_mem_cfg = ttnn.create_sharded_memory_config_(
        shape=(B, 1, H * W, C),
        core_grid=ttnn.CoreGrid(y=8, x=10),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tensor = ttnn.to_memory_config(tensor, new_mem_cfg)  # Issue #39723: Resharding takes absurely long time
    print(f"After reshard - Shape: {tensor.shape}, Layout: {tensor.layout}, Memory: {tensor.memory_config()}")
