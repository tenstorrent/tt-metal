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
@pytest.mark.parametrize("D", [32])
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
