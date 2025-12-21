# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for binary operations with tensors that have identical shapes but different
sharded memory configurations.

This test validates the fix for a bug where tensors with identical shapes but different
sharded memory configurations would incorrectly trigger the native L1 sharding path,
producing incorrect (inf) output. The fix ensures proper memory configuration comparison
before enabling the optimized sharding path.
"""

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp


def test_add_with_mismatched_width_shard_configs(device):
    torch.manual_seed(0)

    tensor_shape = (1, 1, 32, 2048)

    mem_config_a = ttnn.create_sharded_memory_config(
        shape=(32, 64),
        core_grid=ttnn.CoreGrid(x=8, y=4),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    mem_config_b = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=ttnn.CoreGrid(x=8, y=2),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create torch tensors with values in range [-0.5, 0.5] to avoid overflow
    a_torch = torch.rand(tensor_shape, dtype=torch.bfloat16) - 0.5
    b_torch = torch.rand(tensor_shape, dtype=torch.bfloat16) - 0.5

    # Create ttnn tensors with different memory configs
    a = ttnn.from_torch(a_torch, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config_a)
    b = ttnn.from_torch(b_torch, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config_b)

    c_torch = torch.add(a_torch, b_torch)
    c = ttnn.add(a, b, memory_config=mem_config_a)
    c_result = ttnn.to_torch(c)

    assert not torch.isinf(c_result).any(), "Output contains inf values - mismatched shard config bug detected"

    # Verify correctness with PCC and ULP
    assert_with_pcc(c_torch, c_result, 0.9999)
    assert_with_ulp(c_torch, c_result, ulp_threshold=1)
