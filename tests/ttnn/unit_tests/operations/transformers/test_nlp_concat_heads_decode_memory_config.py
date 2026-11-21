# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def _decode_input(device, num_users=32, head_dim=64):
    # Decode layout is [seqlen=1, users, padded heads, head_dim], one HEIGHT_SHARDED core per user.
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
    shard_spec = ttnn.ShardSpec(grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    torch_input = torch.rand(1, num_users, 32, head_dim, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    return torch_input, input_tensor


def test_nlp_concat_heads_decode_honours_memory_config(device):
    # The prim takes memory_config as `/*memory_config*/` and always builds an L1 WIDTH_SHARDED
    # output, so a DRAM request used to be dropped.
    _, input_tensor = _decode_input(device)

    default_output = ttnn.experimental.nlp_concat_heads_decode(input_tensor, num_heads=1)
    dram_output = ttnn.experimental.nlp_concat_heads_decode(
        input_tensor, num_heads=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    assert default_output.memory_config().is_sharded()
    assert dram_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert dram_output.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert torch.equal(ttnn.to_torch(default_output), ttnn.to_torch(dram_output))
