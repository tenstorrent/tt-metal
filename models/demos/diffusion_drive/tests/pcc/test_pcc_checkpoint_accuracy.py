# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

Skips cleanly if the checkpoint or plan-anchor asset is absent, so it is safe in
CI: set ``DD_CHECKPOINT_PATH`` (and ``DD_ANCHOR_PATH``, see conftest) to enable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, load_model
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


def _checkpoint_path() -> str | None:
    data_root = os.environ.get("DD_DATA_ROOT", "/mnt/diffusion-drive")
    candidates = [
        os.environ.get("DD_CHECKPOINT_PATH"),
        f"{data_root}/weights/diffusiondrive_navsim_88p1_PDMS.pth",
    ]
    return next((p for p in candidates if p and Path(p).exists()), None)


@pytest.mark.timeout(1800)
def test_checkpoint_trajectory_pcc(device, model_config) -> None:
    if model_config.plan_anchor_path is None:
        pytest.skip("plan_anchor_path not set — run scripts/prepare_assets.py first")
    ckpt = _checkpoint_path()
    if ckpt is None:
        pytest.skip("real checkpoint not found — set DD_CHECKPOINT_PATH")

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
