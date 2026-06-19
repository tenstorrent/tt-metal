# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Full-model PCC test for the *fully on-device* TtnnDiffusionDriveModel
(Stages 2 + 3 + 3.4 + 3.5 + 3.6 + 3.7) at the production input resolution.

Stage 3.6 puts the ResNet-34 stems and the 4× GPT cross-modal fusion on TTNN, so
the TransFuser backbone runs entirely on-device.  Stage 3.7 puts the AgentHead's
MLPs on TTNN.  Together with 3.4/3.5 (perception head + DDIM denoiser) every
weight-bearing op in the model now executes via TTNN.

The fusion's avg_pool2d / bilinear-upsample ratios are only integer at the
production resolution (camera 256×1024, LiDAR 256×256), so — unlike the reduced
-resolution Stage 2/3/3.4/3.5 tests — this test must run at full resolution.

Validates trajectory + scores + agent outputs PCC ≥ 0.99 vs the Stage-1 reference.
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


@pytest.mark.timeout(1800)
def test_stage3_6_full_on_device_pcc(device, model_config) -> None:
    if model_config.plan_anchor_path is None:
        pytest.skip("plan_anchor_path not set — run scripts/prepare_assets.py first")

    torch.manual_seed(42)
    cfg = DiffusionDriveConfig(plan_anchor_path=model_config.plan_anchor_path, latent=True)
    ref_model = DiffusionDriveModel(cfg).eval()

    # Production resolution — required for the integer pool/upsample ratios in the
    # GPT fusion.  latent=True supplies the 256×256 LiDAR latent; the camera is fed.
    features = {
        "camera_feature": torch.randn(1, 3, 256, 1024),
        "lidar_feature": torch.zeros(1, 1, 256, 256),
        "status_feature": torch.zeros(1, 8),
    }

    torch.manual_seed(1234)  # pin DDIM noise (DD-5)
    with torch.no_grad():
        ref_out = ref_model(features)

    ttnn_model = TtnnDiffusionDriveModel(ref_model, model_config, device)
    (
        ttnn_model.build_stage2(device)
        .build_stage3(device)
        .build_stage3_4(device)
        .build_stage3_5(device)
        .build_stage3_6(device)
        .build_stage3_7(device)
    )

    torch.manual_seed(1234)  # same noise stream
    ttnn_out = ttnn_model(features)

    results = {k: _pcc(ttnn_out[k], ref_out[k]) for k in ("trajectory", "scores", "agent_states", "agent_labels")}
    for k, v in results.items():
        print(f"{k:14s} PCC = {v:.6f}")
    for k, v in results.items():
        assert v >= 0.99, f"{k} PCC {v:.6f} < 0.99 (results: {results})"
