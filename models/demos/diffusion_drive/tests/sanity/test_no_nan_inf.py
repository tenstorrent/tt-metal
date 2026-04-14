# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

from pathlib import Path

import pytest
import torch

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CKPT = _DATA_DIR / "diffusiondrive_navsim.pth"
_ANCHORS = _DATA_DIR / "kmeans_navsim_traj_20.npy"


def _require_assets():
    if not _CKPT.exists() or not _ANCHORS.exists():
        pytest.skip("Assets not found — run scripts/prepare_assets.py first")


def _load_ref_model():
    from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, load_model

    cfg = DiffusionDriveConfig(plan_anchor_path=str(_ANCHORS), latent=True)
    m = load_model(str(_CKPT), cfg, device=torch.device("cpu"))
    m.eval()
    return m


def test_no_nan_inf_output():
    """Forward pass output contains no NaN or Inf."""
    _require_assets()
    torch.manual_seed(42)
    model = _load_ref_model()
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


def test_score_distribution():
    """Scores have non-trivial spread — model is not collapsed to uniform."""
    _require_assets()
    torch.manual_seed(42)
    model = _load_ref_model()
    features = {
        "camera_feature": torch.randn(1, 3, 256, 1024),
        "lidar_feature": torch.zeros(1, 1, 256, 256),
        "status_feature": torch.zeros(1, 8),
    }
    with torch.no_grad():
        out = model(features)

    std = out["scores"].std().item()
    assert std > 1e-3, f"Score std {std:.2e} ≤ 1e-3 — model output collapsed"


def test_trajectory_plausible_range():
    """Trajectory (x, y) positions are within ±100 m (physically plausible)."""
    _require_assets()
    torch.manual_seed(42)
    model = _load_ref_model()
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
