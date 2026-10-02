# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sanity checks for DiffusionDrive TTNN model output.

Verifies:
  - No NaN or Inf in trajectory or scores
  - Score distribution has non-trivial spread (std > 1e-3)
  - Trajectory positions are physically plausible (norm < 100 m)

These tests do not require a device — they run on CPU via the PyTorch
reference model to validate the inference pipeline end-to-end.
"""

from __future__ import annotations

import torch


def _load_ref_model(ckpt: str, anchors: str):
    from models.experimental.diffusion_drive.reference.model import DiffusionDriveConfig, load_model

    cfg = DiffusionDriveConfig(plan_anchor_path=anchors, latent=True)
    m = load_model(ckpt, cfg, device=torch.device("cpu"))
    m.eval()
    return m


def _assets(model_config, checkpoint_path, missing_asset):
    """Resolve both assets through the shared fixtures.

    Skips locally, fails under DD_REQUIRE_ASSETS=1 — the documented CI contract
    (README 7). Resolving repo-local paths directly would skip even when CI has
    staged assets via DD_CHECKPOINT_PATH / DD_ANCHOR_PATH.
    """
    if model_config.plan_anchor_path is None:
        missing_asset("plan_anchor_path not set — run scripts/prepare_assets.py first")
    if checkpoint_path is None:
        missing_asset("real checkpoint not found — run scripts/prepare_assets.py or set DD_CHECKPOINT_PATH")
    return checkpoint_path, model_config.plan_anchor_path


def test_no_nan_inf_output(model_config, checkpoint_path, missing_asset):
    """Forward pass output contains no NaN or Inf."""
    ckpt, anchors = _assets(model_config, checkpoint_path, missing_asset)
    torch.manual_seed(42)
    model = _load_ref_model(ckpt, anchors)
    features = {
        "camera_feature": torch.randn(1, 3, 256, 1024),
        "lidar_feature": torch.zeros(1, 1, 256, 256),
        "status_feature": torch.zeros(1, 8),
    }
    with torch.no_grad():
        out = model(features)

    traj = out["trajectory"]
    scores = out["scores"]

    assert not torch.isnan(traj).any(), "NaN in trajectory"
    assert not torch.isinf(traj).any(), "Inf in trajectory"
    assert not torch.isnan(scores).any(), "NaN in scores"
    assert not torch.isinf(scores).any(), "Inf in scores"


def test_score_distribution(model_config, checkpoint_path, missing_asset):
    """Scores have non-trivial spread — model is not collapsed to uniform."""
    ckpt, anchors = _assets(model_config, checkpoint_path, missing_asset)
    torch.manual_seed(42)
    model = _load_ref_model(ckpt, anchors)
    features = {
        "camera_feature": torch.randn(1, 3, 256, 1024),
        "lidar_feature": torch.zeros(1, 1, 256, 256),
        "status_feature": torch.zeros(1, 8),
    }
    with torch.no_grad():
        out = model(features)

    std = out["scores"].std().item()
    assert std > 1e-3, f"Score std {std:.2e} ≤ 1e-3 — model output collapsed"


def test_trajectory_plausible_range(model_config, checkpoint_path, missing_asset):
    """Trajectory (x, y) positions are within ±100 m (physically plausible)."""
    ckpt, anchors = _assets(model_config, checkpoint_path, missing_asset)
    torch.manual_seed(42)
    model = _load_ref_model(ckpt, anchors)
    features = {
        "camera_feature": torch.randn(1, 3, 256, 1024),
        "lidar_feature": torch.zeros(1, 1, 256, 256),
        "status_feature": torch.zeros(1, 8),
    }
    with torch.no_grad():
        out = model(features)

    traj_xy = out["trajectory"][..., :2]  # B, T, 2
    max_pos = traj_xy.abs().max().item()
    assert max_pos < 100.0, f"Trajectory position {max_pos:.1f} m exceeds 100 m"
