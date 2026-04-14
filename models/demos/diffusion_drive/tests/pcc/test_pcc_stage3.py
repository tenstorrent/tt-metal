# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Full-model PCC test for TtnnDiffusionDriveModel Stage 3.

Validates that after build_stage2() + build_stage3() the trajectory and
scores outputs match the Stage-1 (full-PyTorch) reference with PCC ≥ 0.99.

Stage 3 additionally replaces the 3-level FPN (c5_conv, up_conv5, up_conv4)
with native TTNN conv2d ops; bilinear upsampling stays in PyTorch.

Uses config.latent=True so no real checkpoint is needed.  Camera spatial
size is reduced (64×128) to keep the test fast; lidar resolution stays at
the default 256×256 so the keyval token count (8×8 = 64) is correct.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, DiffusionDriveModel
from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-12:
        return 1.0
    return (a @ b).item() / denom


def _make_model(anchor_path: str) -> DiffusionDriveModel:
    # lidar_resolution defaults to 256×256 so the backbone produces 8×8=64 keyval tokens.
    # latent=True means the lidar_feature input is ignored (learned latent used instead).
    cfg = DiffusionDriveConfig(
        plan_anchor_path=anchor_path,
        latent=True,
    )
    m = DiffusionDriveModel(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_stage3_trajectory_pcc(device, model_config) -> None:
    """Stage-3 trajectory output PCC ≥ 0.99 vs Stage-1 reference."""
    if model_config.plan_anchor_path is None:
        pytest.skip("plan_anchor_path not set — run scripts/prepare_assets.py first")

    torch.manual_seed(42)

    ref_model = _make_model(model_config.plan_anchor_path)

    # Stage 1 forward (pure PyTorch reference)
    features = {
        "camera_feature": torch.randn(1, 3, 64, 128),
        "lidar_feature": torch.zeros(1, 1, 64, 64),
        "status_feature": torch.zeros(1, 8),
    }
    torch.manual_seed(1234)
    with torch.no_grad():
        ref_out = ref_model(features)

    # Stage 3 forward (TTNN backbone + TTNN FPN)
    ttnn_model = TtnnDiffusionDriveModel(ref_model, model_config, device)
    ttnn_model.build_stage2(device).build_stage3(device)

    torch.manual_seed(1234)
    ttnn_out = ttnn_model(features)

    pcc = _pcc(ttnn_out["trajectory"], ref_out["trajectory"])
    assert pcc >= 0.99, f"trajectory PCC {pcc:.6f} < 0.99"


@pytest.mark.timeout(900)
def test_stage3_scores_pcc(device, model_config) -> None:
    """Stage-3 scores output PCC ≥ 0.99 vs Stage-1 reference."""
    if model_config.plan_anchor_path is None:
        pytest.skip("plan_anchor_path not set — run scripts/prepare_assets.py first")

    torch.manual_seed(0)

    ref_model = _make_model(model_config.plan_anchor_path)

    features = {
        "camera_feature": torch.randn(1, 3, 64, 128),
        "lidar_feature": torch.zeros(1, 1, 64, 64),
        "status_feature": torch.zeros(1, 8),
    }
    torch.manual_seed(1234)
    with torch.no_grad():
        ref_out = ref_model(features)

    ttnn_model = TtnnDiffusionDriveModel(ref_model, model_config, device)
    ttnn_model.build_stage2(device).build_stage3(device)

    torch.manual_seed(1234)
    ttnn_out = ttnn_model(features)

    pcc = _pcc(ttnn_out["scores"], ref_out["scores"])
    assert pcc >= 0.99, f"scores PCC {pcc:.6f} < 0.99"
