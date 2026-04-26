# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stage 1 end-to-end PCC test for the full DiffusionDrive model.

Loads the pretrained checkpoint, runs the reference PyTorch model, then runs
the TTNN wrapper, and asserts PCC ≥ 0.99 for trajectory and scores.

Requires:
  - models/demos/diffusion_drive/data/diffusiondrive_navsim.pth
  - models/demos/diffusion_drive/data/kmeans_navsim_traj_20.npy
  Both are downloaded by scripts/prepare_assets.py.

Tests:
  test_full_model_pcc_random    — random inputs, fixed seed
  test_full_model_output_shapes — verify output shapes
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CKPT = _DATA_DIR / "diffusiondrive_navsim.pth"
_ANCHORS = _DATA_DIR / "kmeans_navsim_traj_20.npy"


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-12:
        return 1.0
    return (a @ b).item() / denom


def _require_assets():
    if not _CKPT.exists() or not _ANCHORS.exists():
        pytest.skip("Assets not found — run scripts/prepare_assets.py first")


def _load_ref_model():
    from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, load_model

    cfg = DiffusionDriveConfig(
        plan_anchor_path=str(_ANCHORS),
        latent=True,  # avoid needing real LiDAR sensor data
    )
    model = load_model(str(_CKPT), cfg, device=torch.device("cpu"))
    model.eval()
    return model


def _random_features(batch: int = 1) -> dict:
    torch.manual_seed(42)
    return {
        "camera_feature": torch.randn(batch, 3, 256, 1024),
        "lidar_feature": torch.zeros(batch, 1, 256, 256),
        "status_feature": torch.zeros(batch, 8),
    }


# ---------------------------------------------------------------------------
# Shape test (no device required)
# ---------------------------------------------------------------------------


def test_full_model_output_shapes():
    """Verify model produces (B,8,3) trajectory and (B,20) scores."""
    _require_assets()
    model = _load_ref_model()
    features = _random_features(batch=1)
    with torch.no_grad():
        out = model(features)
    assert out["trajectory"].shape == (1, 8, 3), f"traj shape {out['trajectory'].shape}"
    assert out["scores"].shape == (1, 20), f"scores shape {out['scores'].shape}"


# ---------------------------------------------------------------------------
# PCC test (device required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1])
def test_full_model_pcc_random(device, model_config, batch):
    """
    Stage 1: TTNN wrapper PCC vs PyTorch reference on random inputs.

    Stage 1 uses TorchModuleFallback (full PyTorch forward), so PCC == 1.0.
    The test gates on ≥ 0.99 to be robust to future TTNN op replacements.
    """
    _require_assets()

    from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel

    ref_model = _load_ref_model()
    ttnn_model = TtnnDiffusionDriveModel(ref_model, model_config, device)

    features = _random_features(batch=batch)

    # Reference
    with torch.no_grad():
        ref_out = ref_model(features)

    # TTNN wrapper
    ttnn_out = ttnn_model(features)

    traj_pcc = _pcc(ttnn_out["trajectory"], ref_out["trajectory"])
    scores_pcc = _pcc(ttnn_out["scores"], ref_out["scores"])

    assert traj_pcc >= 0.99, f"trajectory PCC {traj_pcc:.6f} < 0.99"
    assert scores_pcc >= 0.99, f"scores PCC {scores_pcc:.6f} < 0.99"
