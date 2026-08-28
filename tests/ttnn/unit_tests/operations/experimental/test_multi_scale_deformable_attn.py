# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _reference_msda(value: torch.Tensor, grid: torch.Tensor, attn: torch.Tensor, align_corners: bool) -> torch.Tensor:
    """Pure PyTorch reference matching the device op's contract.

    Inputs:
      value: (N, h, w, D)   float32
      grid:  (N, Q*P, 1, 2) float32, normalized to [-1, 1]
      attn:  (N, Q, P)      float32
      align_corners: bilinear pixel-coord mapping (see device-op docstring)

    Output:
      (N, Q, D) float32
    """
    N, h, w, D = value.shape
    Q, P = attn.shape[1], attn.shape[2]

    value_nchw = value.permute(0, 3, 1, 2).contiguous()  # (N, D, h, w)
    grid_for_gs = grid.reshape(N, Q * P, 1, 2)
    sampled = torch.nn.functional.grid_sample(
        value_nchw, grid_for_gs, mode="bilinear", padding_mode="zeros", align_corners=align_corners
    )  # (N, D, Q*P, 1)
    sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (N, Q*P, D)
    sampled = sampled.reshape(N, Q, P, D)
    out = (sampled * attn.unsqueeze(-1)).sum(dim=2)  # (N, Q, D)
    return out


@pytest.mark.parametrize("N", [1, 4])
@pytest.mark.parametrize("h_in,w_in", [(10, 10), (32, 32)])
@pytest.mark.parametrize("D", [16, 32, 48, 64])
@pytest.mark.parametrize("Q", [16, 64])
@pytest.mark.parametrize("P", [4, 8])
@pytest.mark.parametrize("align_corners", [False, True])
def test_msda_correctness(device, N, h_in, w_in, D, Q, P, align_corners):
    torch.manual_seed(0)
    value = torch.randn(N, h_in, w_in, D, dtype=torch.float32)
    grid = torch.rand(N, Q * P, 1, 2, dtype=torch.float32) * 2.0 - 1.0
    attn = torch.softmax(torch.randn(N, Q, P, dtype=torch.float32), dim=-1)

    ref = _reference_msda(value, grid, attn, align_corners=align_corners)

    value_bf = value.to(torch.bfloat16)
    grid_bf = grid.to(torch.bfloat16)
    attn_bf = attn.to(torch.bfloat16)

    value_t = ttnn.from_torch(value_bf, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid_bf, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    attn_t = ttnn.from_torch(attn_bf, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    out_t = ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t, align_corners=align_corners)
    out = ttnn.to_torch(out_t)

    assert_with_pcc(ref, out.to(torch.float32), pcc=0.99)


@pytest.mark.parametrize("D", [8, 24, 40])
def test_msda_rejects_non_multiple_of_16(device, expect_error, D):
    """D values that are not multiples of 16 must be rejected at validation
    with an actionable error, not fail deep in the kernels."""
    N, h_in, w_in, Q, P = 1, 10, 10, 16, 4
    value_t = ttnn.from_torch(
        torch.randn(N, h_in, w_in, D).to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    grid_t = ttnn.from_torch(
        (torch.rand(N, Q * P, 1, 2) * 2.0 - 1.0).to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    attn_t = ttnn.from_torch(
        torch.softmax(torch.randn(N, Q, P), dim=-1).to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    with expect_error(RuntimeError, "multiple of 16"):
        ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t)


@pytest.mark.parametrize("P", [4, 8])
@pytest.mark.parametrize("pts_per_page", [1, 2, 4])
def test_msda_grid_point_folding(device, P, pts_per_page):
    """Every folding of the point axis into the grid's last dimension gives the same answer.

    A ROW_MAJOR page is the last dimension, so (N, Q, 1, P*2) is one NoC read per query where
    (N, Q*P, 1, 2) is P reads of four bytes.
    """
    N, h_in, w_in, D, Q = 2, 16, 16, 32, 32
    torch.manual_seed(0)
    value = torch.randn(N, h_in, w_in, D, dtype=torch.float32)
    grid = torch.rand(N, Q * P, 1, 2, dtype=torch.float32) * 2.0 - 1.0
    attn = torch.softmax(torch.randn(N, Q, P, dtype=torch.float32), dim=-1)

    ref = _reference_msda(value, grid, attn, align_corners=False)

    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_t = ttnn.from_torch(value.to(torch.bfloat16), **rm)
    attn_t = ttnn.from_torch(attn.to(torch.bfloat16), **rm)
    folded = grid.reshape(N, Q * P // pts_per_page, 1, 2 * pts_per_page)
    grid_t = ttnn.from_torch(folded.to(torch.bfloat16), **rm)

    out = ttnn.to_torch(ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t))
    assert_with_pcc(ref, out.to(torch.float32), pcc=0.99)


def test_msda_rejects_grid_width_not_dividing_p(device, expect_error):
    """A grid page must hold a divisor of P points; the reader indexes points modulo it."""
    # 4 points per page against P = 6: the page count still matches Q*P, only the stride does not.
    N, h_in, w_in, D, Q, P = 1, 10, 10, 16, 16, 6
    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_t = ttnn.from_torch(torch.randn(N, h_in, w_in, D).to(torch.bfloat16), **rm)
    grid_t = ttnn.from_torch((torch.rand(N, Q * P // 4, 1, 8) * 2.0 - 1.0).to(torch.bfloat16), **rm)
    attn_t = ttnn.from_torch(torch.softmax(torch.randn(N, Q, P), dim=-1).to(torch.bfloat16), **rm)

    with expect_error(RuntimeError, "divisor of P"):
        ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t)
