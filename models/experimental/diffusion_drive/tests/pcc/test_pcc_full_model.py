# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end PCC test for the full DiffusionDrive model on random inputs.

Loads the pretrained checkpoint, runs the reference PyTorch model, then runs the
**fully built** TTNN model (``build_all``) and asserts PCC >= 0.99 for trajectory
and scores.

Building the stack matters: an unbuilt ``TtnnDiffusionDriveModel`` leaves
``_perception`` as None and ``__call__`` falls through to the CPU reference, so the
comparison would be the reference against itself — PCC 1.0 that proves nothing
about TTNN. Complements test_pcc_checkpoint_accuracy (real-checkpoint accuracy)
and test_pcc_stage4 (per-stage outputs) by covering the whole graph at once.

Assets resolve through the shared ``checkpoint_path`` / ``model_config`` fixtures;
a missing one skips locally but fails under ``DD_REQUIRE_ASSETS=1`` (README 7).

Tests:
  test_full_model_pcc_random    — random inputs, fixed seed, full TTNN stack
  test_full_model_output_shapes — verify reference output shapes
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc


def _assets(model_config, checkpoint_path, missing_asset):
    if model_config.plan_anchor_path is None:
        missing_asset("plan_anchor_path not set — run scripts/prepare_assets.py first")
    if checkpoint_path is None:
        missing_asset("real checkpoint not found — run scripts/prepare_assets.py or set DD_CHECKPOINT_PATH")
    return checkpoint_path, model_config.plan_anchor_path


def _load_ref_model(ckpt: str, anchors: str):
    from models.experimental.diffusion_drive.reference.model import DiffusionDriveConfig, load_model

    cfg = DiffusionDriveConfig(
        plan_anchor_path=anchors,
        latent=True,  # avoid needing real LiDAR sensor data
    )
    model = load_model(ckpt, cfg, device=torch.device("cpu"))
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


def test_full_model_output_shapes(model_config, checkpoint_path, missing_asset):
    """Verify model produces (B,8,3) trajectory and (B,20) scores."""
    ckpt, anchors = _assets(model_config, checkpoint_path, missing_asset)
    model = _load_ref_model(ckpt, anchors)
    features = _random_features(batch=1)
    with torch.no_grad():
        out = model(features)
    assert out["trajectory"].shape == (1, 8, 3), f"traj shape {out['trajectory'].shape}"
    assert out["scores"].shape == (1, 20), f"scores shape {out['scores'].shape}"


# ---------------------------------------------------------------------------
# PCC test (device required)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
@pytest.mark.parametrize("batch", [1])
def test_full_model_pcc_random(device, model_config, checkpoint_path, missing_asset, batch):
    """Fully built TTNN model vs the PyTorch reference, random inputs."""
    ckpt, anchors = _assets(model_config, checkpoint_path, missing_asset)

    from models.experimental.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel

    ref_model = _load_ref_model(ckpt, anchors)
    features = _random_features(batch=batch)

    # Run the PyTorch reference BEFORE building the TTNN wrapper: TtnnDiffusionDriveModel
    # keeps `ref_model` as its `_model` and build_all() swaps its backbone, perception and
    # trajectory modules in place, so a reference forward taken afterwards already runs TTNN.
    torch.manual_seed(1234)  # pin DDIM noise (README 3.5)
    with torch.no_grad():
        ref_out = ref_model(features)

    ttnn_model = TtnnDiffusionDriveModel(ref_model, model_config, device)
    ttnn_model.build_all(device)
    assert ttnn_model._perception is not None, "build_all did not install the TTNN perception path"

    torch.manual_seed(1234)  # same noise stream
    ttnn_out = ttnn_model(features)

    traj_pcc = comp_pcc(ref_out["trajectory"], ttnn_out["trajectory"])[1]
    scores_pcc = comp_pcc(ref_out["scores"], ttnn_out["scores"])[1]
    print(f"full-model trajectory PCC = {traj_pcc:.6f}, scores PCC = {scores_pcc:.6f}")

    assert traj_pcc >= 0.99, f"trajectory PCC {traj_pcc:.6f} < 0.99"
    assert scores_pcc >= 0.99, f"scores PCC {scores_pcc:.6f} < 0.99"
