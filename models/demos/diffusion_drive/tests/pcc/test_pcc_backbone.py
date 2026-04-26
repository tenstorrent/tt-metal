# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PCC tests for TtnnTransfuserBackbone (DiffusionDrive Stage 2).

Validates that the TTNN backbone (ResNet-34 BasicBlock stages in TTNN,
stem + GPT + FPN in PyTorch) matches the reference PyTorch backbone with
PCC ≥ 0.99 on both output tensors:
  • bev_upscale  — (B, 64, H_bev, W_bev)
  • bev_feature  — (B, 512, H_lid/32, W_lid/32)

Inputs are deliberately down-scaled to keep the test under 5 minutes:
  camera  (1, 3,  64, 128)  → layer4 spatial:  (2,  4)
  lidar   (1, 1,  64,  64)  → layer4 spatial:  (2,  2)

The backbone is instantiated with config.latent=True so no real LiDAR
checkpoint is required.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, TransfuserBackbone
from models.demos.diffusion_drive.tt.ttnn_backbone import TtnnTransfuserBackbone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-12:
        return 1.0
    return (a @ b).item() / denom


def _make_backbone(latent: bool = True) -> TransfuserBackbone:
    """Build a TransfuserBackbone with random weights (no checkpoint needed)."""
    # Use a minimal anchor path (plan_anchor_path not needed for backbone test)
    cfg = DiffusionDriveConfig(
        plan_anchor_path=None,
        latent=latent,
        lidar_resolution_height=64,
        lidar_resolution_width=64,
    )
    bb = TransfuserBackbone(cfg)
    bb.eval()
    return bb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(600)
def test_backbone_bev_upscale_pcc(device) -> None:
    """bev_upscale output: TtnnTransfuserBackbone vs reference PCC ≥ 0.99."""
    torch.manual_seed(42)

    bb_ref = _make_backbone(latent=True)
    ttnn_bb = TtnnTransfuserBackbone(bb_ref, device)

    # Small camera input; lidar is unused (latent=True)
    camera = torch.randn(1, 3, 64, 128)
    lidar_dummy = torch.zeros(1, 1, 64, 64)

    with torch.no_grad():
        ref_up, ref_feat, _ = bb_ref(camera, lidar_dummy)
        ttnn_up, ttnn_feat, _ = ttnn_bb(camera, lidar_dummy)

    pcc_up = _pcc(ttnn_up, ref_up)
    assert pcc_up >= 0.99, f"bev_upscale PCC {pcc_up:.6f} < 0.99"


@pytest.mark.timeout(600)
def test_backbone_bev_feature_pcc(device) -> None:
    """bev_feature output: TtnnTransfuserBackbone vs reference PCC ≥ 0.99."""
    torch.manual_seed(7)

    bb_ref = _make_backbone(latent=True)
    ttnn_bb = TtnnTransfuserBackbone(bb_ref, device)

    camera = torch.randn(1, 3, 64, 128)
    lidar_dummy = torch.zeros(1, 1, 64, 64)

    with torch.no_grad():
        _, ref_feat, _ = bb_ref(camera, lidar_dummy)
        _, ttnn_feat, _ = ttnn_bb(camera, lidar_dummy)

    pcc_feat = _pcc(ttnn_feat, ref_feat)
    assert pcc_feat >= 0.99, f"bev_feature PCC {pcc_feat:.6f} < 0.99"
