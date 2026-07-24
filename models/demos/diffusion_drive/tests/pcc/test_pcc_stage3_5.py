# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Full-model PCC test for TtnnDiffusionDriveModel Stage 3.5.

After build_stage2/3/3_4/3_5, the TrajectoryHead DDIM denoiser's compute also
runs on TTNN (plan_anchor_encoder, time_mlp, and the 2-layer CustomTransformer
decoder: grid-sample cross-attention, both MHAs, FFN, norms, FiLM modulation,
task heads).  Validates trajectory + scores PCC ≥ 0.99 vs the Stage-1 reference.
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
    cfg = DiffusionDriveConfig(plan_anchor_path=anchor_path, latent=True)
    return DiffusionDriveModel(cfg).eval()


@pytest.mark.timeout(900)
@pytest.mark.parametrize("output_key", ["trajectory", "scores"])
def test_stage3_5_pcc(device, model_config, output_key) -> None:
    if model_config.plan_anchor_path is None:
        pytest.skip("plan_anchor_path not set — run scripts/prepare_assets.py first")

    torch.manual_seed(42)
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
    ttnn_model.build_stage2(device).build_stage3(device).build_stage3_4(device).build_stage3_5(device)

    torch.manual_seed(1234)
    ttnn_out = ttnn_model(features)

    pcc = _pcc(ttnn_out[output_key], ref_out[output_key])
    assert pcc >= 0.99, f"{output_key} PCC {pcc:.6f} < 0.99"
