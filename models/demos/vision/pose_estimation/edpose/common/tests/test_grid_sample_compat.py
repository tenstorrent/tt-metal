# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 0: grid_sample compatibility test for ED-Pose on TT hardware.

Validates that ttnn.grid_sample produces correct results for the exact tensor
shapes used inside ED-Pose's Multi-Scale Deformable Attention.

ED-Pose profile (800x1216 input, Swin-L 5scale):
  - Encoder: input=[8, 32, H_l, W_l], grid=[8, 80997, 4, 2]  (per level, 6 layers)
  - Decoder (box):  input=[8, 32, H_l, W_l], grid=[8, 900, 4, 2]
  - Decoder (pose): input=[8, 32, H_l, W_l], grid=[8, 1800, 4, 2]

  Feature map spatial sizes (5 levels):
    Level 0: 200x304, Level 1: 100x152, Level 2: 50x76, Level 3: 25x38, Level 4: 13x19

  ttnn grid_sample expects NHWC layout:
    input: (N, H, W, C)  grid: (N, H_out, W_out, 2)
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def pytorch_grid_sample_nhwc(input_nhwc, grid, align_corners=False):
    """Reference grid_sample operating on NHWC tensors."""
    input_nchw = input_nhwc.permute(0, 3, 1, 2).contiguous()
    output_nchw = F.grid_sample(
        input_nchw.float(), grid.float(), mode="bilinear", padding_mode="zeros", align_corners=align_corners
    )
    return output_nchw.permute(0, 2, 3, 1).contiguous()


# Decoder shapes — smaller grid, feasible for initial testing
DECODER_SHAPES = [
    # (N, C, H_in, W_in, H_grid, W_grid) — grid is (N, H_grid, W_grid, 2)
    # Decoder box queries: grid = [8, 900, 4, 2] → reshaped to (8, 900, 4, 2)
    pytest.param(8, 32, 200, 304, 900, 4, id="decoder_box_level0"),
    pytest.param(8, 32, 100, 152, 900, 4, id="decoder_box_level1"),
    pytest.param(8, 32, 50, 76, 900, 4, id="decoder_box_level2"),
    pytest.param(8, 32, 25, 38, 900, 4, id="decoder_box_level3"),
    pytest.param(8, 32, 13, 19, 900, 4, id="decoder_box_level4"),
    # Decoder pose queries: grid = [8, 1800, 4, 2]
    pytest.param(8, 32, 200, 304, 1800, 4, id="decoder_pose_level0"),
    pytest.param(8, 32, 100, 152, 1800, 4, id="decoder_pose_level1"),
    pytest.param(8, 32, 50, 76, 1800, 4, id="decoder_pose_level2"),
    pytest.param(8, 32, 25, 38, 1800, 4, id="decoder_pose_level3"),
    pytest.param(8, 32, 13, 19, 1800, 4, id="decoder_pose_level4"),
]

# Encoder shapes — large grid (80997 query points), may stress memory
ENCODER_SHAPES = [
    pytest.param(8, 32, 200, 304, 80997, 4, id="encoder_level0"),
    pytest.param(8, 32, 100, 152, 80997, 4, id="encoder_level1"),
    pytest.param(8, 32, 50, 76, 80997, 4, id="encoder_level2"),
    pytest.param(8, 32, 25, 38, 80997, 4, id="encoder_level3"),
    pytest.param(8, 32, 13, 19, 80997, 4, id="encoder_level4"),
]

# Small shapes for quick sanity checks
SMALL_SHAPES = [
    pytest.param(1, 32, 8, 8, 16, 4, id="small_1"),
    pytest.param(2, 32, 16, 16, 64, 4, id="small_2"),
    pytest.param(8, 32, 25, 38, 100, 4, id="small_3"),
]


@pytest.mark.parametrize("N, C, H_in, W_in, H_grid, W_grid", SMALL_SHAPES)
def test_grid_sample_small_shapes(device, N, C, H_in, W_in, H_grid, W_grid):
    """Quick sanity check with small tensors."""
    torch.manual_seed(42)

    input_nhwc = torch.randn(N, H_in, W_in, C, dtype=torch.bfloat16)
    grid = (torch.rand(N, H_grid, W_grid, 2, dtype=torch.float32) * 2.0 - 1.0)

    ref_output = pytorch_grid_sample_nhwc(input_nhwc, grid)

    tt_input = ttnn.from_torch(input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32)

    tt_output = ttnn.grid_sample(tt_input, tt_grid, mode="bilinear", align_corners=False)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(ref_output.to(torch.bfloat16), tt_output_torch, pcc=0.98)


@pytest.mark.parametrize("N, C, H_in, W_in, H_grid, W_grid", DECODER_SHAPES)
def test_grid_sample_decoder_shapes(device, N, C, H_in, W_in, H_grid, W_grid):
    """Test with ED-Pose decoder tensor shapes (900/1800 query points)."""
    torch.manual_seed(42)

    input_nhwc = torch.randn(N, H_in, W_in, C, dtype=torch.bfloat16)
    grid = (torch.rand(N, H_grid, W_grid, 2, dtype=torch.float32) * 2.0 - 1.0)

    ref_output = pytorch_grid_sample_nhwc(input_nhwc, grid)

    tt_input = ttnn.from_torch(input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32)

    tt_output = ttnn.grid_sample(tt_input, tt_grid, mode="bilinear", align_corners=False)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(ref_output.to(torch.bfloat16), tt_output_torch, pcc=0.98)


@pytest.mark.parametrize("N, C, H_in, W_in, H_grid, W_grid", ENCODER_SHAPES)
def test_grid_sample_encoder_shapes(device, N, C, H_in, W_in, H_grid, W_grid):
    """Test with ED-Pose encoder tensor shapes (80997 query points).

    These are the largest shapes and may require significant device memory.
    Mark as slow/nightly if they exceed single-device capacity.
    """
    torch.manual_seed(42)

    input_nhwc = torch.randn(N, H_in, W_in, C, dtype=torch.bfloat16)
    grid = (torch.rand(N, H_grid, W_grid, 2, dtype=torch.float32) * 2.0 - 1.0)

    ref_output = pytorch_grid_sample_nhwc(input_nhwc, grid)

    tt_input = ttnn.from_torch(input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32)

    tt_output = ttnn.grid_sample(tt_input, tt_grid, mode="bilinear", align_corners=False)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(ref_output.to(torch.bfloat16), tt_output_torch, pcc=0.98)
