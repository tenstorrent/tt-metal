# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import verify_with_pcc


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "H, W, number_of_blocks",
    (
        (32, 32, 1),
        (64, 32, 2),
        (576, 64, 3),
    ),
)
def test_unet_dram_sharded_weights(H, W, number_of_blocks, dtype, device, use_program_cache, reset_seeds):
    torch_weight = torch.randn(1, 1, H, W, dtype=torch.bfloat16)
    ttnn_weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    [shard_height, number_of_cores] = unet_shallow_ttnn.determine_num_cores_for_dram_sharded_weights(
        H // number_of_blocks, max_cores=8, minimum_shard_size=32
    )

    def f(weight, num_blocks, num_cores):
        _, _, H, W = weight.shape
        block_size = H // num_blocks
        weight = weight.reshape(num_blocks, num_cores, block_size // num_cores, -1)
        weight = weight.permute(1, 0, 2, 3)
        weight = weight.reshape(1, 1, H, W)
        return weight

    expected = f(torch_weight, number_of_blocks, number_of_cores)
    actual = ttnn.to_torch(
        unet_shallow_ttnn.group_weight_blocks_per_core(ttnn_weight, number_of_blocks, number_of_cores)
    )

    verify_with_pcc(expected, actual, 1 if dtype == ttnn.bfloat16 else 0.9999)
