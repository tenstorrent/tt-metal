# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Conv2dConfig.diagonal_weight_senders (BLOCK_SHARDED): the weight multicast senders move from one row / column of
cores to a diagonal, with one writer kernel for senders and receivers. The result must be bit-identical to the
default layout, on the transposed (COL_MAJOR) and the ROW_MAJOR block-sharded paths."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# name: (H = W, C_in, C_out, grid (x, y), shard [rows, cols], transpose_shards, act_block_h_override)
CASES = {
    # SDXL full-grid resnet conv shapes: 1024 rows over 11 columns (ragged), 4096 rows over 11 columns
    "t1024_1280": (32, 1280, 1280, (11, 10), [96, 128], True, 0),
    "t4096_640": (64, 640, 640, (11, 10), [384, 64], True, 0),
    "rm4096_640": (64, 640, 640, (10, 10), [416, 64], False, 0),
}


def _run(device, case, diagonal):
    hw, cin, cout, (gx, gy), shard, transpose, abh = CASES[case]
    torch.manual_seed(0)
    x = torch.randn(1, cin, hw, hw)
    w = torch.randn(cout, cin, 3, 3) / (9 * cin) ** 0.5
    b = torch.randn(cout) * 0.1
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    orient = ttnn.ShardOrientation.COL_MAJOR if transpose else ttnn.ShardOrientation.ROW_MAJOR
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard, orient)
    )
    tx = ttnn.from_torch(
        x.permute(0, 2, 3, 1).reshape(1, 1, hw * hw, cin),
        ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        deallocate_activation=True,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        act_block_h_override=abh,
        transpose_shards=transpose,
        diagonal_weight_senders=diagonal,
    )
    out = ttnn.conv2d(
        input_tensor=tx,
        weight_tensor=ttnn.from_torch(w, ttnn.bfloat16),
        bias_tensor=ttnn.from_torch(b.reshape(1, 1, 1, -1), ttnn.bfloat16),
        in_channels=cin,
        out_channels=cout,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=hw,
        input_width=hw,
        conv_config=conv_config,
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        ),
        dtype=ttnn.bfloat16,
    )
    result = ttnn.to_torch(out).float()[..., :cout].reshape(1, hw, hw, cout).permute(0, 3, 1, 2)
    ttnn.deallocate(out)
    return torch.nn.functional.conv2d(x, w, b, padding=1), result


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_diagonal_weight_senders(device, case):
    gx, gy = CASES[case][3]
    grid = device.compute_with_storage_grid_size()
    if grid.x < gx or grid.y < gy:
        pytest.skip(f"needs a {gx}x{gy} worker grid, device has {grid.x}x{grid.y}")
    ref, line = _run(device, case, diagonal=False)
    _, diag = _run(device, case, diagonal=True)
    assert_with_pcc(ref, diag, 0.99)
    assert torch.equal(line, diag), "diagonal weight senders changed the result"


def test_diagonal_weight_senders_config():
    cfg = ttnn.Conv2dConfig()
    assert cfg.diagonal_weight_senders is False
    assert ttnn.Conv2dConfig(diagonal_weight_senders=True).diagonal_weight_senders is True
