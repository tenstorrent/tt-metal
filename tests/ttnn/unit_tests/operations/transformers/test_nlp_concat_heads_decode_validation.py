# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def _decode_input(device, head_dim, num_users=32):
    # Decode layout is [seqlen=1, users, padded heads, head_dim], one HEIGHT_SHARDED core per user.
    padded_head_dim = (head_dim + 31) // 32 * 32
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
    shard_spec = ttnn.ShardSpec(grid, [32, padded_head_dim], ttnn.ShardOrientation.ROW_MAJOR)
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    torch_input = torch.rand(1, num_users, 32, head_dim, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    return torch_input, tt_input


def test_nlp_concat_heads_decode_rejects_unaligned_head_dim(device, expect_error):
    # The hidden dim and the per-head tile stride come from the padded last dim, so a logical head dim
    # of 48 (padded 64) came back as 64 columns per head instead of 48.
    _, tt_input = _decode_input(device, head_dim=48)

    with expect_error(RuntimeError, "logical head dim"):
        ttnn.experimental.nlp_concat_heads_decode(tt_input, num_heads=1)


def test_nlp_concat_heads_decode_aligned_head_dim_control(device):
    # Control: a tile-aligned head dim still concatenates exactly.
    torch_input, tt_input = _decode_input(device, head_dim=64)

    output = ttnn.experimental.nlp_concat_heads_decode(tt_input, num_heads=1)

    expected = torch_input[:, :, :1].reshape(1, 1, 32, 64)
    assert torch.equal(expected, ttnn.to_torch(output))
