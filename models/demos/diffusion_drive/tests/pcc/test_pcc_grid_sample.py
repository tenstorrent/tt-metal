# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for the TTNN GridSampleCrossBEVAttention port (Stage-5 de-risk).

Proves that ``ttnn.grid_sample`` is a working drop-in for the ``F.grid_sample``
deformable BEV sampling that the README listed as the key Stage-5 blocker.

Two checks:
  test_grid_sample_primitive_pcc   — ttnn.grid_sample vs F.grid_sample at the
                                     model's exact shapes (C=256, BEV 64×64,
                                     K=20, T=8).
  test_grid_sample_attention_pcc   — full GridSampleCrossBEVAttention block
                                     output, TTNN vs reference.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, GridSampleCrossBEVAttention
from models.demos.diffusion_drive.tt.ttnn_grid_sample_attention import TtnnGridSampleCrossBEVAttention


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-12:
        return 1.0
    return (a @ b).item() / denom


@pytest.mark.timeout(300)
def test_grid_sample_primitive_pcc(device) -> None:
    """ttnn.grid_sample matches F.grid_sample (bilinear/zeros/align_corners=False)."""
    import ttnn

    torch.manual_seed(0)
    B, C, H, W, K, T = 1, 256, 64, 64, 20, 8
    value = torch.randn(B, C, H, W)
    grid = torch.rand(B, K, T, 2) * 2 - 1

    ref = F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    v_tt = ttnn.from_torch(
        value.permute(0, 2, 3, 1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    g_tt = ttnn.from_torch(grid.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.grid_sample(v_tt, g_tt)
    o = ttnn.to_torch(out).reshape(B, K, T, C).permute(0, 3, 1, 2)

    pcc = _pcc(o, ref)
    assert pcc >= 0.99, f"grid_sample primitive PCC {pcc:.6f} < 0.99"


@pytest.mark.timeout(300)
def test_grid_sample_attention_pcc(device) -> None:
    """Full GridSampleCrossBEVAttention block: TTNN (grid_sample on device) vs reference."""
    torch.manual_seed(0)
    cfg = DiffusionDriveConfig(plan_anchor_path=None)  # config only used for lidar_max_x/y

    B, K, T, D = 1, 20, 8, 256
    H = W = 64
    ref_mod = GridSampleCrossBEVAttention(embed_dims=D, num_heads=8, num_points=T, config=cfg, in_bev_dims=256).eval()

    queries = torch.randn(B, K, D)
    traj_points = torch.randn(B, K, T, 2) * 10.0  # ego-metres, within ±32 grid
    bev_feature = torch.randn(B, 256, H, W)

    with torch.no_grad():
        ref_out = ref_mod(queries, traj_points, bev_feature, (H, W))

    ttnn_mod = TtnnGridSampleCrossBEVAttention(ref_mod, device)
    with torch.no_grad():
        ttnn_out = ttnn_mod(queries, traj_points, bev_feature, (H, W))

    pcc = _pcc(ttnn_out, ref_out)
    assert pcc >= 0.99, f"grid_sample attention PCC {pcc:.6f} < 0.99"
