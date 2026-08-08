# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the GroupNorm wrapper (models/common/group_norm.py) on main.

Validates all negative_mask modes on a block-sharded ROW_MAJOR input:
  - small/auto      : fits -> auto picks the no-negative-mask path
  - big/auto        : L1-tight -> auto picks the negative-mask path (via calc or safety-net fallback)
  - big/forced_neg  : negative_mask=True
  - small/forced_no : negative_mask=False
Each asserts PCC vs torch and that the wrapper took the expected path (gn.last_decision).
"""
import pytest
import torch

import ttnn

from models.common.group_norm import GroupNorm


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _sharded_input(device, N, C, H, W, grid):
    torch_input = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    inp = torch_input.permute(0, 2, 3, 1).reshape(N, 1, W * H, C)
    inp = ttnn.from_torch(inp, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_coord = ttnn.CoreCoord(grid.x - 1, grid.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = (N * H * W // grid.y, C // grid.x)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec)
    return torch_input, ttnn.to_device(inp, device, memory_config=mem), mem


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "C, H, W, neg, expect_neg_used",
    [
        (640, 64, 64, "auto", False),  # fits -> no negative mask
        (640, 128, 128, "auto", True),  # L1-tight -> negative mask
        (640, 128, 128, True, True),  # forced on
        (640, 64, 64, False, False),  # forced off
    ],
    ids=["small_auto", "big_auto", "big_forced_neg", "small_forced_no"],
)
def test_gn_module(device, C, H, W, neg, expect_neg_used):
    if device.core_grid.y == 7:
        pytest.skip()
    torch.manual_seed(0)
    N, num_groups = 1, 32
    grid = ttnn.CoreGrid(y=8, x=8)

    torch_input, inp, mem = _sharded_input(device, N, C, H, W, grid)
    w = torch.rand((C,), dtype=torch.bfloat16)
    b = torch.rand((C,), dtype=torch.bfloat16)

    gn = GroupNorm(
        num_channels=C,
        num_groups=num_groups,
        core_grid=grid,
        device=device,
        weight=w,
        bias=b,
        negative_mask=neg,
    )

    out = gn(inp, memory_config=mem)
    out_t = ttnn.to_torch(ttnn.from_device(ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)))

    ref = torch.nn.functional.group_norm(torch_input, num_groups, weight=w, bias=b)
    ref = ref.permute(0, 2, 3, 1).reshape(N, 1, W * H, C)

    pcc = _pcc(ref, out_t)
    print(f"\n[gn-module] C={C} H={H} W={W} neg={neg!r}  last_decision={gn.last_decision}  PCC={pcc:.5f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
    assert (
        gn.last_decision[1] == expect_neg_used
    ), f"expected negative_mask_used={expect_neg_used}, got {gn.last_decision}"


def test_gn_module_rejects_negative_mask_without_input_mask(expect_error):
    # The negative-mask path needs an input mask to overlap; building it without one must fail
    # clearly at construction (not crash later on device). No device needed — the guard is first.
    grid = ttnn.CoreGrid(y=8, x=8)
    for mode in (True, "auto"):
        with expect_error(ValueError, "requires use_input_mask=True"):
            GroupNorm(
                num_channels=640,
                num_groups=32,
                core_grid=grid,
                device=None,
                negative_mask=mode,
                use_input_mask=False,
            )
