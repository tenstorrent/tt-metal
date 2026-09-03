# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Production-resolution accuracy gate with the REAL trained checkpoint.

Unlike test_pcc_stage2/3/4 (random weights, ``latent=True`` — op-equivalence
only), this loads the trained 88.x checkpoint (``latent=False``) and asserts the
full on-device stack (``build_stage2``..``build_stage4``) matches the PyTorch
reference trajectory at production resolution (camera 256×1024, LiDAR 256×256).
It is the committed analogue of ``scripts/navsim_inproc/check_parity.py`` (which
measured trajectory PCC 0.999705), closing the "only random-weight gates are
committed" gap.

Asset handling is deliberately two-mode.  Locally the assets are optional and the
test skips, so a checkout without a 730 MB checkpoint is still usable.  In CI the
job stages the assets (``scripts/prepare_assets.py``) and sets
``DD_REQUIRE_ASSETS=1``, which turns every "asset missing" path into a **failure**
— a silently skipped gate is not a gate.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.diffusion_drive.reference.model import DiffusionDriveConfig, load_model
from models.experimental.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel


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
def test_checkpoint_trajectory_pcc(device, model_config, checkpoint_path, missing_asset) -> None:
    if model_config.plan_anchor_path is None:
        missing_asset("plan_anchor_path not set — run scripts/prepare_assets.py first")
    ckpt = checkpoint_path
    if ckpt is None:
        missing_asset("real checkpoint not found — run scripts/prepare_assets.py or set DD_CHECKPOINT_PATH")

    # Real trained weights, latent=False (the deployed eval config).
    ref_cfg = DiffusionDriveConfig(plan_anchor_path=model_config.plan_anchor_path, latent=False)
    ref_model = load_model(ckpt, ref_cfg, device=torch.device("cpu")).eval()

    features = {
        "camera_feature": torch.randn(1, 3, 256, 1024),
        "lidar_feature": torch.randn(1, 1, 256, 256),
        "status_feature": torch.randn(1, 8),
    }

    torch.manual_seed(1234)  # pin DDIM noise (DD-5)
    with torch.no_grad():
        ref_out = ref_model(features)

    # Build the full on-device stack on the (now captured) reference model.
    ttnn_model = TtnnDiffusionDriveModel(ref_model, model_config, device)
    (
        ttnn_model.build_stage2(device)
        .build_stage3(device)
        .build_stage3_4(device)
        .build_stage3_5(device)
        .build_stage3_6(device)
        .build_stage3_7(device)
        .build_stage4(device)
    )

    torch.manual_seed(1234)  # same noise stream
    ttnn_out = ttnn_model(features)

    pcc = _pcc(ttnn_out["trajectory"], ref_out["trajectory"])
    print(f"checkpoint trajectory PCC = {pcc:.6f}")
    assert pcc >= 0.99, f"checkpoint trajectory PCC {pcc:.6f} < 0.99"
