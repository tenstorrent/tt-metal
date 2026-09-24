# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Operand-rank and dtype plumbing in the binary_ng and ternary factories (#57356)."""

import pytest
import torch
import ttnn


def _tt(x, device, dtype=ttnn.bfloat16, **kwargs):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kwargs)


def _q(x, device):
    return ttnn.to_torch(_tt(x, device)).float()


# extract_nD_dims walked the lower-rank operand down to the output rank and indexed past its own
# shape, so a rank-7 operand broadcast against a rank-6 one threw instead of running.
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [((1, 2, 3, 4, 5, 32, 32), (2, 3, 4, 5, 32, 32)), ((2, 3, 4, 5, 32, 32), (1, 2, 3, 4, 5, 32, 32))],
)
def test_add_rank7_against_rank6(device, a_shape, b_shape):
    torch.manual_seed(0)
    a, b = torch.rand(a_shape), torch.rand(b_shape)
    out = ttnn.to_torch(ttnn.add(_tt(a, device), _tt(b, device))).float()
    # Same op with the lower-rank operand expanded by hand: same kernel math, so bit-identical.
    out_shape = torch.broadcast_shapes(a.shape, b.shape)
    expected = ttnn.to_torch(
        ttnn.add(_tt(a.expand(out_shape).contiguous(), device), _tt(b.expand(out_shape).contiguous(), device))
    ).float()
    assert torch.equal(out, expected)


def test_where_rank7_against_rank6(device):
    torch.manual_seed(0)
    c = (torch.rand(1, 2, 3, 4, 5, 32, 32) > 0.5).float()
    a, b = torch.rand(2, 3, 4, 5, 32, 32), torch.rand(2, 3, 4, 5, 32, 32)
    out = ttnn.to_torch(ttnn.where(_tt(c, device), _tt(a, device), _tt(b, device))).float()
    assert torch.equal(out, torch.where(c.bool(), _q(a, device), _q(b, device)))


# adjust_to_shape aligned the two shapes from the left, folding a lower-rank input's width into its
# height volume, so the derived output shard was too short to hold the broadcast output.
def test_where_sharded_lower_rank_condition(device):
    torch.manual_seed(0)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    cond_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )
    c = (torch.rand(256, 64) > 0.5).float()
    a, b = torch.rand(1, 4, 256, 64), torch.rand(1, 4, 256, 64)
    out = ttnn.to_torch(ttnn.where(_tt(c, device, memory_config=cond_mc), _tt(a, device), _tt(b, device))).float()
    assert torch.equal(out, torch.where(c.bool(), _q(a, device), _q(b, device)))


# A UINT32 lerp used to reach the LLK with an undeclared fill function and fail to JIT-compile; with
# that fixed it still returns zeros, since the LERP kernel has no unsigned path. It is rejected instead.
def test_lerp_rejects_uint32(device, expect_error):
    a = _tt(torch.randint(0, 100, (1, 1, 64, 64), dtype=torch.int32), device, dtype=ttnn.uint32)
    b = _tt(torch.randint(0, 100, (1, 1, 64, 64), dtype=torch.int32), device, dtype=ttnn.uint32)
    with expect_error(RuntimeError, "lerp does not support UINT32 inputs"):
        ttnn.lerp(a, b, 0.5)
    with expect_error(RuntimeError, "lerp does not support UINT32 inputs"):
        ttnn.lerp(a, b, b)
