# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_for_blackhole
from ttnn.operations.block_sharded_rms_norm import block_sharded_rms_norm


@skip_for_blackhole("Generic-op block-sharded RMS norm is not validated on Blackhole")
@pytest.mark.parametrize("num_cols", [1, 2], ids=["single_col", "multi_col"])
def test_block_sharded_rms_norm(device, num_cols):
    grid = device.compute_with_storage_grid_size()
    if grid.x < num_cols:
        pytest.skip(f"Need at least {num_cols} cores across x, have {grid.x}")

    num_rows = 2 if grid.y >= 2 else 1
    height_per_core = 64
    width_per_core = 64
    height = height_per_core * num_rows
    width = width_per_core * num_cols
    epsilon = 1e-5

    torch.manual_seed(0)
    x = torch.randn((1, 1, height, width), dtype=torch.bfloat16)

    mem_config = ttnn.create_sharded_memory_config(
        x.shape,
        ttnn.CoreGrid(x=num_cols, y=num_rows),
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    input_tensor = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )

    output_tensor = block_sharded_rms_norm(input_tensor, epsilon=epsilon)
    output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_torch = ttnn.to_torch(output_tensor)

    expected = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    assert torch.allclose(output_torch.float(), expected, rtol=1e-2, atol=1e-2)
