# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("hw", [1024])
def test_unary_i0_op(hw, device):
    # tor_a = torch.tensor([[ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]], dtype=torch.bfloat16)
    tor_a = torch.randn(1, 1, hw, hw, dtype=torch.bfloat16)

    tor_res = torch.i0(tor_a)
    mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.L1, shard_spec=None
    )

    tt_a = ttnn.from_torch(tor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)

    result = ttnn.i0(tt_a)

    tt_res = ttnn.to_torch(result)
    # print("tt_res", tt_res)
    # print("tor_res", tor_res)
    # print("abs_diff", torch.abs(tt_res - tor_res))

    # assert torch.equal(tt_res, tor_res)

    pcc, pcc_msg = assert_with_pcc(tor_res, tt_res, 0.999)
    assert pcc


@pytest.mark.parametrize("hw", [1024])
def test_unary_i0_shard(hw, device):
    # tor_a = torch.tensor([[ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]], dtype=torch.bfloat16)
    tor_a = torch.randn(1, 1, hw, hw, dtype=torch.bfloat16)

    tor_res = torch.i0(tor_a)
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange((0, 0), (7, 7)),
        }
    )
    shard_spec = ttnn.ShardSpec(shard_grid, [128, 128], ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardMode.PHYSICAL)
    mem = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec)

    tt_a = ttnn.from_torch(tor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)

    result = ttnn.i0(tt_a)

    tt_res = ttnn.to_torch(result)
    # print("tt_res", tt_res)
    # print("tor_res", tor_res)
    # print("abs_diff", torch.abs(tt_res - tor_res))

    # assert torch.equal(tt_res, tor_res)

    pcc, pcc_msg = assert_with_pcc(tor_res, tt_res, 0.999)
    assert pcc
